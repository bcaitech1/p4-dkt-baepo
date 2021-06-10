import enum
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from pytz import timezone
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dkt.dataloader import (DatasetArgs, Preprocess, PreprocessArgs,
                            generate_dataset, get_loaders)
from dkt.model import LSTMATTN, DktBert, DktNewBert
import dkt.trainer as base_trainer

# AdamW와 Cosine Anealing을 같이 써보고 Test Error를 다음 링크와 비교
# https://medium.com/vitalify-asia/whats-up-with-deep-learning-optimizers-since-adam-5c1d862b9db0


logger = logging.getLogger(__name__)

@dataclass
class Path:
    training_data: str
    test_data: str
    checkpoint: str
    tensorboard: str
    log: str


class TraineeBase:
    def __init__(
        self,
        trainee_name: str,
        data_paths: Dict[str, str],
        save_paths: Dict[str, str],
        hyperparameters: Dict,
        device: torch.device,
    ) -> None:
        logger.info("Initializing trainee...")

        self.name = trainee_name
        self.device = device
        self.hyperparameters = hyperparameters
        
        training_path = os.path.join(data_paths['train_root_dir'], data_paths['training_data_file'])
        test_path = os.path.join(data_paths['test_root_dir'], data_paths['test_data_file'])
        checkpoint_path = os.path.join(save_paths['root_dir'], save_paths['checkpoints_dir'], self.name)
        tensorboard_path = os.path.join(save_paths['root_dir'], save_paths['tensorboard_dir'], self.name)
        log_path = os.path.join(save_paths['root_dir'], save_paths['yaml_dir'], self.name)

        self.path = Path(
            training_path, test_path,
            checkpoint_path, tensorboard_path, log_path
        )

        seed_everything(hyperparameters['seed'])

    def train(self):
        print()
        logger.info(f"{self.name}({type(self).__name__}) Traininig start...")

        self.on_train_begin()
        self.on_train()
        self.on_train_end()
        
        logger.info("Traininig finished")

    def on_train_begin(self):
        now = datetime.now(timezone('Asia/Seoul'))
        current_time = now.strftime("%y%m%d_%H%M%S")
        self.tensorboard = SummaryWriter(log_dir=os.path.join(self.path.tensorboard, current_time))
        logger.info(f"Tensorboard initialized [{self.tensorboard.log_dir}]")

    def on_train(self):
        raise NotImplementedError("Train function not implemented")

    def on_train_end(self):
        pass


class DktBertTrainee(TraineeBase):
    def __init__(
        self,
        trainee_name: str,
        data_paths: Dict,
        save_paths: Dict,
        hyperparameters: Dict,
        device: torch.device
    ) -> None:
        super().__init__(trainee_name, data_paths, save_paths, hyperparameters, device)
        self.logging_step = 50

        logger.info("Loading training data...")
        
        args = PreprocessArgs(
            asset_dir=data_paths['asset_dir'],
            data_dir=data_paths['train_root_dir']
        )
        preprocess = Preprocess(args)
        preprocess.load_train_data(data_paths['training_data_file'])
        train_data, self.dataset_attr = preprocess.get_train_data()
        # self.dataset_attr은 args 안 쓰기 위해 사용하는 변수...
        
        self.train_dataset, self.valid_dataset = preprocess.split_data(train_data)
        logger.info("Loading data finished!")
    
    def on_train(self):
        dataset_args = DatasetArgs(**self.hyperparameters['dataloader'])
        train_loader, valid_loader = get_loaders(dataset_args, self.train_dataset, self.valid_dataset)

        epochs = self.hyperparameters['epochs']
        total_steps = int(len(train_loader.dataset) / dataset_args.batch_size) * epochs
        warmup_steps = total_steps // 10
        # warmup_steps는 linear_schedule_with_warmup에서만 사용된다.
        # 현재는 pleatu만 사용하므로 사용 안됨

        # 모델 불러오기
        model_args: Dict = self.hyperparameters['model']['args']
        model_args['device'] = self.device
        model_args['max_seq_len'] = dataset_args.max_seq_len
        model_args.update(self.dataset_attr)
        model = DktBert(**model_args)
        model.to(self.device)

        # optimzier 및 scheduler
        optimizer = AdamW(model.parameters(), **self.hyperparameters['optimizer']['args'])
        scheduler = ReduceLROnPlateau(optimizer, **self.hyperparameters['scheduler']['args'])
        # 주의 optimizer-name, scheduler-name 두 속성을 무시하고 있음
        # 이름에 따라 로드하는 부분이 구현 안 됨
        
        best_auc = -1
        early_stopping_counter = 0
        for epoch in range(epochs):

            print(f"Start Training: Epoch {epoch + 1}")
            
            ### TRAIN
            train_auc, train_acc, train_loss = base_trainer.train(
                train_loader, model, optimizer, 
                self.hyperparameters['clip_grad'], 
                self.logging_step, 
                self.device
            )
            
            ### VALID
            auc, acc, _, _ = base_trainer.validate(valid_loader, model, self.device)
            
            self.tensorboard.add_scalar('Valid/train_auc', train_auc, epoch)
            self.tensorboard.add_scalar('Valid/train_acc', train_acc, epoch)
            self.tensorboard.add_scalar('Valid/valid_auc', auc, epoch)
            self.tensorboard.add_scalar('Valid/valid_acc', acc, epoch)

            ### model save or early stopping
            if auc > best_auc:
                best_auc = auc
                # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
                model_to_save = model.module if hasattr(model, 'module') else model
                base_trainer.save_checkpoint({   # Best 모델 저장
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    },
                    self.path.checkpoint, 'best_model.pt',
                )
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.hyperparameters['patience']:
                    print(f'Early Stopping counter: {early_stopping_counter} out of {self.hyperparameters["patience"]}')
                    break

            # scheduler
            scheduler.step(best_auc)

            base_trainer.save_checkpoint({       # 마지막 모델 저장
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                }, 
                self.path.checkpoint, 
                'last_model.pt'
            )

        # Train 끝


class DktLSTMAttnTrainee(TraineeBase):
    def __init__(
        self,
        trainee_name: str,
        data_paths: Dict,
        save_paths: Dict,
        hyperparameters: Dict,
        device: torch.device
    ) -> None:
        super().__init__(trainee_name, data_paths, save_paths, hyperparameters, device)
        self.logging_step = 50

        logger.info("Loading training data...")
        args = PreprocessArgs(
            asset_dir=data_paths['asset_dir'],
            data_dir=data_paths['train_root_dir']
        )
        preprocess = Preprocess(args)
        preprocess.load_train_data(data_paths['training_data_file'])
        train_data, self.dataset_attr = preprocess.get_train_data()
        # self.dataset_attr은 args 안 쓰기 위해 사용하는 변수...
        
        self.train_dataset, self.valid_dataset = preprocess.split_data(train_data)

        logger.info("Loading data finished!")
    
    def on_train(self):
        dataset_args = DatasetArgs(**self.hyperparameters['dataloader'])
        train_loader, valid_loader = get_loaders(dataset_args, self.train_dataset, self.valid_dataset)

        epochs = self.hyperparameters['epochs']
        total_steps = int(len(train_loader.dataset) / dataset_args.batch_size) * epochs
        warmup_steps = total_steps // 10
        # warmup_steps는 linear_schedule_with_warmup에서만 사용된다.
        # 현재는 pleatu만 사용하므로 사용 안됨

        # 모델 불러오기
        model_args: Dict = self.hyperparameters['model']
        model_args['device'] = self.device
        model_args.update(self.dataset_attr)
        model = LSTMATTN(**model_args)
        model.to(self.device)

        # optimzier 및 scheduler
        optimizer = AdamW(model.parameters(), **self.hyperparameters['optimizer']['args'])
        scheduler = ReduceLROnPlateau(optimizer, **self.hyperparameters['scheduler']['args'])
        # 주의 optimizer-name, scheduler-name 두 속성을 무시하고 있음
        # 이름에 따라 로드하는 부분이 구현 안 됨
        
        best_auc = -1
        early_stopping_counter = 0
        for epoch in range(epochs):

            print(f"Start Training: Epoch {epoch + 1}")
            
            ### TRAIN
            train_auc, train_acc, train_loss = base_trainer.train(
                train_loader, model, optimizer, 
                self.hyperparameters['clip_grad'], 
                self.logging_step, 
                self.device
            )
            
            ### VALID
            auc, acc, _, _ = base_trainer.validate(valid_loader, model, self.device)
            
            self.tensorboard.add_scalar('Valid/train_auc', train_auc, epoch)
            self.tensorboard.add_scalar('Valid/train_acc', train_acc, epoch)
            self.tensorboard.add_scalar('Valid/valid_auc', auc, epoch)
            self.tensorboard.add_scalar('Valid/valid_acc', acc, epoch)

            ### model save or early stopping
            if auc > best_auc:
                best_auc = auc
                # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
                model_to_save = model.module if hasattr(model, 'module') else model
                base_trainer.save_checkpoint({   # Best 모델 저장
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    },
                    self.path.checkpoint, 'best_model.pt',
                )
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.hyperparameters['patience']:
                    print(f'Early Stopping counter: {early_stopping_counter} out of {self.hyperparameters["patience"]}')
                    break

            # scheduler
            scheduler.step(best_auc)

            base_trainer.save_checkpoint({       # 마지막 모델 저장
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                }, 
                self.path.checkpoint, 
                'last_model.pt'
            )


class DktNewBertTrainee(TraineeBase):
    def __init__(
        self,
        trainee_name: str,
        data_paths: Dict,
        save_paths: Dict,
        hyperparameters: Dict,
        device: torch.device
    ) -> None:
        super().__init__(trainee_name, data_paths, save_paths, hyperparameters, device)
        self.logging_step = 20

        logger.info("Loading training data...")

        self.dataset_folds = generate_dataset(
            self.path.training_data,
            data_paths['asset_dir'],
            valid_split=True, 
            **self.hyperparameters['dataset'])
        logger.info("Loading data finished!")
    
    def on_train(self):
        for index, dataset_fold in enumerate(self.dataset_folds):
            if self.hyperparameters['target'] >= 0 and index != self.hyperparameters['target']:
                continue

            train_dataset, valid_dataset = dataset_fold
            train_loader = DataLoader(train_dataset, collate_fn=train_dataset.collate, **self.hyperparameters['dataloader']['train_args'])
            valid_loader = DataLoader(valid_dataset, collate_fn=valid_dataset.collate, **self.hyperparameters['dataloader']['valid_args'])

            model_args: Dict = self.hyperparameters['model']['args']
            feature_info = train_dataset.features     # config에 따라 모델이 달라진다?
            model = DktNewBert(
                feature_info=feature_info,
                max_seq_len=train_dataset.max_seq_len,
                **model_args
            )
            # print(model)
            model.to(self.device)
            optimizer = AdamW(model.parameters(), **self.hyperparameters['optimizer']['args'])
            # scheduler = ReduceLROnPlateau(optimizer, **self.hyperparameters['scheduler']['args'])
            scheduler = CosineAnnealingWarmRestarts(optimizer, **self.hyperparameters['scheduler']['args'])
            
            best_auc = -1
            early_stopping_counter = 0
            epochs: int = self.hyperparameters['epochs']
            for epoch in range(epochs):
                logger.info(f'Epoch [{epoch + 1} / {epochs}] start')
                ### TRAIN   추후 train 코드를 수정할 필요가 있으면 수정한다.
                train_auc, train_acc, train_loss = train(
                    train_loader, model, optimizer, 
                    self.hyperparameters['clip_grad'], 
                    self.logging_step, 
                    self.device,
                    self.tensorboard,
                    epoch
                )
                
                ### VALID   마찬가지로 validate 코드를 수정할 필요가 있으면 수정한다.
                auc, acc, _, _ = validate(valid_loader, model, self.device)
                
                self.tensorboard.add_scalar('Valid/train_auc', train_auc, epoch)
                self.tensorboard.add_scalar('Valid/train_acc', train_acc, epoch)
                self.tensorboard.add_scalar('Valid/valid_auc', auc, epoch)
                self.tensorboard.add_scalar('Valid/valid_acc', acc, epoch)

                ### model save or early stopping
                if auc > best_auc:
                    best_auc = auc
                    # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
                    model_to_save = model.module if hasattr(model, 'module') else model
                    save_checkpoint({   # Best 모델 저장
                        'epoch': epoch + 1,
                        'state_dict': model_to_save.state_dict(),
                        },
                        self.path.checkpoint, 'best_model.pt',
                    )
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= self.hyperparameters['patience']:
                        print(f'Early Stopping counter: {early_stopping_counter} out of {self.hyperparameters["patience"]}')
                        break

                # scheduler
                scheduler.step(best_auc)

                save_checkpoint({       # 마지막 모델 저장
                        'epoch': epoch + 1,
                        'state_dict': model_to_save.state_dict(),
                    }, 
                    self.path.checkpoint, 
                    'last_model.pt'
                )

            # 각 Fold Train 끝
        # Fold 전체 Train 끝

# ------------ Train 관련 ------------ #

def train(
    train_loader: DataLoader, model, optimizer: AdamW,
    clip_grad: int,
    logging_step: int,
    device: torch.device,
    tensorboard: SummaryWriter,
    current_epoch: int,
):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        input_batch, targets = process_batch(batch, device)
        preds = model(input_batch)

        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, clip_grad)

        if step % logging_step == 0:
            print(f"Training steps: {step} / {len(train_loader)} Loss: {str(loss.item())}")
            tensorboard.add_scalar('Train/loss', loss.item(), step * (current_epoch + 1))

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        if device.type == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else:  # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses)/len(losses)
    print(f'TRAIN AUC : {auc} ACC : {acc}')
    return auc, acc, loss_avg


def validate(valid_loader, model, device: torch.device):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        input_batch, targets = process_batch(batch, device)

        # Forward
        preds = model(input_batch)

        # predictions
        preds = preds[:,-1]
        targets = targets[:,-1]
    
        if device.type == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    
    print(f'VALID AUC : {auc} ACC : {acc}\n')

    return auc, acc, total_preds, total_targets


# 배치 전처리 (수정 해야함)
def process_batch(batch: Tuple[torch.Tensor], device: torch.device):

    mask = batch[-1].to(torch.float)
    answer = batch[-2].to(torch.float)
    features = batch[:-2]

    new_features: List[torch.Tensor] = []
    for feature in features:
        new_features.append(
            ((feature + 1) * mask.to(feature.dtype))
        )   # continous 관계 없이 1을 더해버리는데 이 부분 고민해보기

    # device memory로 이동
    new_features.append(mask)
    result = []
    for feature in new_features:
        result.append(feature.to(device))
    answer = answer.to(device)
    
    return tuple(result), answer


# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = nn.BCELoss(reduction="none")
    loss = loss(preds, targets)
    #마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:,-1]
    loss = torch.mean(loss)
    return loss


def get_metric(targets, preds):
    auc = roc_auc_score(targets, preds)
    acc = accuracy_score(targets, np.where(preds >= 0.5, 1, 0))

    return auc, acc


def update_params(loss, model, optimizer, clip_grad):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state, model_dir, model_filename):
    print('saving model ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    torch.save(state, os.path.join(model_dir, model_filename))








# ---------------------------------

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
