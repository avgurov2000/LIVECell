import os, sys
import streamlit as st
from print_result import read_image_plot, read_image, ONNXModel, plot_results

import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SCRIPT_DIR)
from model import list2dict, LightningMRCNN
from dataset import resize

IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pics")
MODEL_PATH = os.path.join(SCRIPT_DIR, "models_arch/epoch=19-step=11000.onnx")
CKPT_PATH = os.path.join(SCRIPT_DIR, "models_arch/epoch=19-step=11000.ckpt") 
IMG_FORMATS = ".jpg", ".jpeg", ".png", ".tif", ".tf"

def get_prediction_ckpt(predictions, confidence: float = 0.65):
    scores = predictions[0]["scores"].detach().cpu().numpy()
    scores = list(scores[scores >= confidence])
    pred_t = len(scores)
    
    boxes = [(i[0], i[1], i[2], i[3]) for ii, i in enumerate(predictions[0]["boxes"].detach().cpu().numpy()) if ii < pred_t]
    masks = predictions[0]['masks'].squeeze().detach().cpu().numpy()[:pred_t]
    return boxes, masks


def get_coloured_mask_ckpt(mask):
    mask_bool = mask > 0
    colours = [0, 0, 255]
    r = np.zeros_like(mask_bool).astype(np.uint8)
    g = np.zeros_like(mask_bool).astype(np.uint8)
    b = np.zeros_like(mask_bool).astype(np.uint8)
    r[mask_bool == 1], g[mask_bool == 1], b[mask_bool == 1] = colours
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


@torch.no_grad()
def plot_results_ckpt(model: nn.Module, image_path: str, confidence: float = 0.5):
    model.eval()
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    resize_shape = image.shape
    
    image = resize(image=image)
    
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    image = torch.as_tensor(image).float()/255
    
    prediction = model([image], None)
    
    image = image.squeeze().detach().cpu().numpy()
    image = (image*255).astype(np.uint8)
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    
    boxes, masks = get_prediction_ckpt(prediction)
    rect_th=2
    text_size=2
    text_th=2
    for i in range(len(masks)):
        rgb_mask = get_coloured_mask_ckpt(masks[i])
        image = cv2.addWeighted(image, 1, rgb_mask, 0.3, 0)
        
    image = resize(image=image, size=resize_shape)
    return image



def main():
    global model_onnx, model_ckpt, checkpoint
    
    st.title("LIVECell Instance Segmentation")
    pics = os.listdir(IMAGE_PATH)
    
    img_path = st.sidebar.selectbox("Select image", pics)
    img_path = os.path.join(IMAGE_PATH, img_path)
    image = read_image_plot(img_path)
    st.image(img_path, width=700)
    
    clicked1 = st.button("Segmentize (onnx)")
    clicked2 = st.button("Segmentize (ckpt)")
    
    if clicked1:
        model_onnx = ONNXModel(MODEL_PATH) if model_onnx is None else model_onnx
        image_for_process = read_image(img_path)
        result = plot_results(model_onnx, image_for_process)
        st.image(result, width=700)
        
    if clicked2:
        model_ckpt = LightningMRCNN("efficientnet-b0", params, torch.device("cpu")) if model_ckpt is None else model_ckpt
        checkpoint = torch.load(CKPT_PATH) if checkpoint is None else checkpoint
        model_ckpt.load_state_dict(checkpoint["state_dict"])
        model_ckpt.eval()
        result = Image.fromarray(plot_results_ckpt(model_ckpt, img_path))
        st.image(result, width=700)
        
if __name__ == "__main__":
    
    model_onnx = None
    model_ckpt = None
    checkpoint = None
    params = {"in_channels": 3, "pyramid_channels": 64, "depth": 5, "num_classes": 2, "lr0": 0.005}

    main()