import os, sys
import torch
import cv2
import numpy as np
import tqdm

import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union, Tuple
from PIL import Image

import onnx
import onnxruntime

def read_image_plot(path:str, size=()):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(image)
    return image

def read_image(path:str, size=()):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (352, 260))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    image = np.asarray(image).astype(np.float32)/255
    return image



class ONNXModel():
    
    def __init__(self, path: str):
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        self.ort_session = onnxruntime.InferenceSession(path)
        self.return_names = ['boxes', 'labels', 'scores', 'masks']
        
    def __call__(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        
        ort_inputs = {
            self.ort_session.get_inputs()[0].name: data,
        }
        
        ort_outs = self.ort_session.run(None, ort_inputs)
        named_outs = {name: value for name, value in zip(self.return_names, ort_outs)}
        
        return named_outs
    
    
def process_predictions(
    prediction: Dict[str, np.ndarray],
    confidence: float = 0.3
) -> Tuple[np.ndarray]:
    
    scores = prediction["scores"]
    boxes = prediction["boxes"]
    masks = prediction["masks"]
    
    scores_conf = list(scores[scores >= confidence])
    size = len(scores_conf)
    
    boxes_conf = boxes[:size]
    masks_conf = masks[:size]
    return boxes_conf, masks_conf


def get_coloured_mask(mask):
    colours = [0, 0, 255]
    mask_bool = mask>0
    r = np.zeros_like(mask_bool).astype(np.uint8)
    g = np.zeros_like(mask_bool).astype(np.uint8)
    b = np.zeros_like(mask_bool).astype(np.uint8)
    r[mask_bool == 1], g[mask_bool == 1], b[mask_bool == 1] = colours
    coloured_mask = np.stack([r, g, b], axis=-1).squeeze()
    return coloured_mask


def plot_results(model: ONNXModel, image: np.ndarray, confidence: float = 0.5, resize=(704, 520)):
    
    pred = model(image)
    image = cv2.cvtColor((image.squeeze()*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    boxes, masks = process_predictions(pred)
        
    rect_th=2
    text_size=2
    text_th=2
    for i in range(len(masks)):
        rgb_mask = get_coloured_mask(masks[i])
        image = cv2.addWeighted(image, 1, rgb_mask, 0.3, 0)
        cv2.rectangle(image, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])), [255, 0, 0], rect_th)
        
    image = cv2.resize(image, resize)
    return Image.fromarray(image)