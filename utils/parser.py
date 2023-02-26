import argparse
import yaml
import io

import torch
from torch import nn

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="resnet18", help='model name')
    parser.add_argument('--config', type=str, default="", help='config .yaml file')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--device', default='cpu', help='cuda (e.g. cuda:0, cuda:1, etc) or cpu')
    parser.add_argument('--seed', type=int, default=2023, help='Global training seed')
    parser.add_argument('--num_workers', type=int, default=2, help='data processing workers count')
    
    return parser.parse_args()

def yaml_opt(path):
    
    default_data = {
        "path": r"",
        "train_dir": r"",
        "validation_dir": r"",
        "project": "cell_segmentation",
        "version": "0",
        "hyperparameters": 
            {
                "in_channels": 3, 
                "pyramid_channels": 128, 
                "depth": 5, 
                "num_classes": 2, 
                "lr0": 0.001
        }
    }
    
    with open(path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        
    for key in default_data:
        if key not in data_loaded:
            data_loaded[key] = default_data[key]
    
    return data_loaded
    

if __name__ == "__main__":
    opt = parse_opt()
    
    config_file = opt.config
    model_name = opt.model
    batch_size = opt.batch_size
    epochs = opt.epochs
    device = torch.device(opt.device)
    seed = opt.seed
    name = opt.name
    num_workers = opt.num_workers
    
    parameters = yaml_opt(config_file)
    print(parameters)
