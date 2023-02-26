import warnings
warnings.filterwarnings('ignore')

import shutil
import os, sys
from model import list2dict, LightningMRCNN


import torch
from torch import nn
import cv2
import numpy as np

from utils.parser import yaml_opt
import argparse

import onnx
import onnxruntime
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="efficientnet-b0", help='model name')
    parser.add_argument('--config', type=str, default="./configs/config_final.yaml", help='config .yaml file')
    parser.add_argument('--ckpt_file', type=str, default="./runs/cell_segmentation_a6/efficientnet-b0_v0_test/checkpoints/epoch=19-step=11000.ckpt", help='model checkpoint file (.ckpt)')
    parser.add_argument('--save_path', type=str, default="./models_arch", help='model save path')
    return parser.parse_args()


if __name__ == "__main__":
    
    
    ### Parse parameters
    opt = parse_opt()
    
    model_name = opt.model
    config_file = opt.config
    ckpt_file = opt.ckpt_file    
    save_path = opt.save_path
    parameters = yaml_opt(config_file)
    
    
    model = LightningMRCNN(model_name, parameters["hyperparameters"], torch.device("cpu"))
    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    x = [torch.rand(1, 260, 352, requires_grad=True)]
    torch_out = model.model(x)
    
    model_save_name = os.path.splitext(ckpt_file.split(os.sep)[-1])[0]
    model_save_name = os.path.join(save_path, model_save_name)
    torch.onnx.export(
        model, 
        (x, None), 
        f"{model_save_name}.onnx", 
        export_params=True, 
        opset_version=13, 
        do_constant_folding=True,
        input_names = ['input1', 'input2'], 
        output_names = ['output']
    )
    
    

    onnx_model = onnx.load(f"{model_save_name}.onnx")
    onnx.checker.check_model(onnx_model)
    shutil.copy(ckpt_file, f"{model_save_name}.ckpt")
