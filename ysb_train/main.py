import argparse
import logging
import os
from typing import Dict, Type

import torch
import yaml

import dkt.train
from dkt.inference import predict, predict_new_dkt_bert
from dkt.train import TraineeBase

logging.basicConfig(
    format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S', 
    level=logging.INFO
)

logger = logging.getLogger(__name__)


def load_config(config_file_name: str):
    training_config = {}
    with open(f"./{config_file_name}", 'r', encoding='utf8') as fr:
        training_config = yaml.load(fr, Loader=yaml.FullLoader)
        logger.info("config.yaml loaded")

    return training_config


def initialize_folders(save_paths: Dict[str, str]):
    root_path = save_paths["root_dir"]
    if not os.path.isdir(root_path):
        os.mkdir(root_path)
        logger.info(f"{root_path} created")

    for key in save_paths:
        if key != "root_dir":
            target_path = os.path.join(root_path, save_paths[key])

            if not os.path.isdir(target_path):
                os.mkdir(target_path)
                logger.info(f"{target_path} created")


if __name__ == "__main__":
    print()
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=str, default="True")
    parser.add_argument("--evaluation", type=str, default="True")
    parser.add_argument("--config_file", type=str, default="config.yaml")
    option_args = parser.parse_args()

    do_training: bool = option_args.training.lower() == "true"
    do_inference: bool = option_args.evaluation.lower() == "true"
    logger.info(option_args)

    # Get target device
    logger.info(f"PyTorch version: [{torch.__version__}]")
    if torch.cuda.is_available():   # 무조건 cuda만 사용
        target_device = torch.device("cuda:0")
    else:
        raise Exception("No CUDA Device")
    logger.info(f"Target device: [{target_device}]")

    # Load config file
    config = load_config(option_args.config_file)
    config["device"] = target_device

    # 필요한 폴더 생성
    initialize_folders(config["save_paths"])
    
    # Training
    if do_training:
        trainee_class: Type[TraineeBase] = getattr(dkt.train, config["trainee_type"])
        del config["trainee_type"]

        trainee = trainee_class(**config)
        trainee.train()
        
    if do_inference:
        # trainee_type에 따라 predict할 수 있도록 수정할 것 
        # (현재는 LSTM-Attention 전용)
        if 'trainee_type' in config:
            del config["trainee_type"]
        predict_new_dkt_bert(**config)
    
    logger.info("Process Finished!")
            
