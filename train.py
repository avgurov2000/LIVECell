import warnings
warnings.filterwarnings('ignore')

import os, sys
import random
import math
import numpy as np
import torch

from dataset import CellDataset
from model import LightningMRCNN
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Union, Optional
from utils.parser import parse_opt, yaml_opt

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks


import tqdm
import cv2


def collate_fn(data):

    images = []
    targets = []
    for im, tg in data:
        images.append(im)
        targets.append(tg)

    return images, targets
    
def print_name(name):
    char_count = len(name)*5
    chr_count_middle = char_count - len(name) - 2
    chr_count_left = chr_count_middle//2
    chr_count_right = chr_count_middle - chr_count_left
    
    print("#"*char_count)
    print("#"*chr_count_left+" "+name+" "+"#"*chr_count_right)
    print("#"*char_count)
   

def collate_fn(data):
    mages = []
    targets = []
    for im, tg in data:
        images.append(im)
        targets.append(tg)

    return images, targets
    
if __name__ == "__main__":
    
    
    ### Parse parameters
    opt = parse_opt()
    
    model_name = opt.model
    config_file = opt.config
    batch_size = opt.batch_size
    epochs = opt.epochs
    device = torch.device(opt.device)
    seed = opt.seed
    num_workers = opt.num_workers
    
    parameters = yaml_opt(config_file)
    
    
    print_name(model_name)
    
    

    train = CellDataset(
        path=parameters["train_dir"], 
        data_path=parameters["path"], 
        augmentations=True #False
    )

    test = CellDataset(
        path=parameters["validation_dir"], 
        data_path=parameters["path"], 
        augmentations=False
    )
    
   


    
    project = parameters["project"]
    version = str(parameters["version"])
    print(f"Project: {project}, model: {model_name}, version: {version}")
    
    config = {
        "model_name": model_name,
        "batch_size": batch_size,
        "epochs": epochs,
        "hyperparameters": parameters["hyperparameters"]
    }
    
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device != torch.device("cpu"):
        torch.set_float32_matmul_precision('medium')
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
        
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers
    )
      

    model = LightningMRCNN(model_name, parameters["hyperparameters"], device)
    
    logger = pl_loggers.WandbLogger(save_dir="runs", name=model_name + "_v" + version, project=project, version=model_name + "_v" + version, config=config)
    checkpoint_callback  = callbacks.ModelCheckpoint(save_last = True, monitor="map_50", mode="max")
    
    try:
        trainer = pl.Trainer(accelerator="cpu" if device=="cpu" else "gpu", devices=1, max_epochs=epochs, logger=logger, callbacks=[checkpoint_callback])
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
        torch.cuda.empty_cache()

    except Exception as e:
        print(e)
        torch.cuda.empty_cache()
        del model

