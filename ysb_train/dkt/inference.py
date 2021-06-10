import logging
import os
from typing import Any, Dict

import torch
from numpy import ndarray
from torch.utils.data import DataLoader
from tqdm import tqdm

from dkt.dataloader import (DatasetArgs, Preprocess, PreprocessArgs,
                            generate_dataset, get_loaders)
from dkt.model import DktNewBert, LSTMATTN, DktBert
from dkt.train import Path, process_batch
# from dkt.trainer import process_batch

logger = logging.getLogger(__name__)


def predict(
        trainee_name: str,
        data_paths: Dict,
        save_paths: Dict,
        hyperparameters: Dict,
        device: torch.device
    ):
    logger.info('Loading test data...')
    args = PreprocessArgs(
        asset_dir=data_paths['asset_dir'],
        data_dir=data_paths['test_root_dir']
    )
    preprocess = Preprocess(args)
    preprocess.load_test_data(data_paths['test_data_file'])
    test_data, dataset_attr = preprocess.get_test_data()

    dataset_args = DatasetArgs(**hyperparameters['dataloader'])
    logger.info('Loading data finished!')
    
    hyperparameters['model']['args'].update(dataset_attr)
    predict_dkt_bert(
        os.path.join(save_paths['root_dir'], save_paths['checkpoints_dir'], trainee_name, 'best_model.pt'),
        save_paths['root_dir'],
        test_data,
        hyperparameters,
        dataset_args,
        device
    )

def predict_new_dkt_bert(
        trainee_name: str,
        data_paths: Dict,
        save_paths: Dict,
        hyperparameters: Dict,
        device: torch.device
    ):
    training_path = os.path.join(data_paths['train_root_dir'], data_paths['training_data_file'])
    test_path = os.path.join(data_paths['test_root_dir'], data_paths['test_data_file'])
    checkpoint_path = os.path.join(save_paths['root_dir'], save_paths['checkpoints_dir'], trainee_name)
    tensorboard_path = os.path.join(save_paths['root_dir'], save_paths['tensorboard_dir'], trainee_name)
    log_path = os.path.join(save_paths['root_dir'], save_paths['yaml_dir'], trainee_name)

    path = Path(
        training_path, test_path,
        checkpoint_path, tensorboard_path, log_path
    )

    logger.info("Loading test data...")
    dataset_folds = generate_dataset(
        path.test_data,
        data_paths['asset_dir'],
        valid_split=False, 
        **hyperparameters['dataset'])
    logger.info("Loading data finished!")
    test_dataset = dataset_folds[0][0]
    
    test_loader = DataLoader(test_dataset, collate_fn=test_dataset.collate, **hyperparameters['dataloader']['test_args'])
    
    # Model Loading
    model_path = os.path.join(checkpoint_path, 'best_model.pt')
    logger.info(f'Loading Model from: {model_path}')
    load_state = torch.load(model_path)
    
    feature_info = test_dataset.features

    model = DktNewBert(
        feature_info=feature_info,
        max_seq_len= test_dataset.max_seq_len,
        **hyperparameters['model']['args']
    )
    # model = DktBert(**model_args)
    # model = LSTMATTN(**model_args)
    model.to(device)

    model.load_state_dict(load_state['state_dict'], strict=True)
    logger.info("Loading Model Finished.")

    logger.info('Evaluating...')
    total_preds = []
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        input_batch, targets = process_batch(batch, device)
        preds: torch.Tensor = model(input_batch)

        # predictions
        preds = preds[:, -1]
        # 혹시 에러날 경우 target의 -1 레이블을 확인해 볼 것
        
        if device.type != 'cpu':
            preds = preds.to('cpu').detach().numpy()
        else:
            preds = preds.detach().numpy()
            
        total_preds += preds.tolist()
    
    logger.info('Evaluation finished')

    output_dir = save_paths['root_dir']
    write_path = os.path.join(output_dir, "output.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))
    
    logger.info(f'Prediction file saved at {write_path}')

def predict_dkt_bert(
        model_path: str,
        output_dir: str,
        test_data: ndarray,
        hyperparameters: Dict, 
        dataset_args: DatasetArgs,
        device: torch.device,
    ):
    logger.info('Evaluating...')

    model_args = hyperparameters['model']['args']
    model_args['device'] = device
    # model_args['max_seq_len'] = dataset_args.max_seq_len
    model = load_model(model_path, model_args, device)
    model.eval()
    _, test_loader = get_loaders(dataset_args, None, test_data)
    
    total_preds = []
    
    for step, batch in enumerate(test_loader):
        input = process_batch(batch, device)

        preds = model(input)
        
        # predictions
        preds = preds[:,-1]
        
        if device.type == 'cuda':
            preds = preds.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            
        total_preds+=list(preds)

    logger.info('Evaluation finished')

    write_path = os.path.join(output_dir, "output.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))
    
    logger.info(f'Prediction file saved at {write_path}')

def load_model(
        model_path: str, 
        model_args: Dict,
        device: torch.device
    ):   
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)

    # Model
    # model = DktBert(**model_args)
    model = LSTMATTN(**model_args)
    model.to(device)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)
   
    
    print("Loading Model from:", model_path, "...Finished.")
    return model
