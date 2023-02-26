"""
Image augmentation functions
"""

import math
import random
import albumentations as A

import torchvision
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from typing import List, Dict, Tuple, Union, Optional

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation

def resize(image: np.ndarray, size=(260,352)):
    T_resize = [A.Resize(*size)]
    resize = A.Compose(T_resize)
    return resize(image=image)["image"]

class Albumentations:
    """
        Class-transformer for data augmentations (including images, bboxes, segmentation masks)

            Attributes:
                transform (Optional[object]) - data transformer albumentations.Compose class
                resize (object) - data resize albumentations.Compose class
                enable (bool) - flag responcible for whether to apply augmentations to data or not
    """
    
    def __init__(self, enable: bool) -> None:
        self.transform = None
        self.enable = enable
        """
        Class-transformer constructor

            Attributes:
                transform (Optional[object]) - data transformer albumentations.Compose class
                enable (bool) - flag responcible for whether to apply augmentations to data or not
        """
        try:
            import albumentations as A
            T_resize = [A.Resize(260,352)] #520,704
            T = [
                    A.HorizontalFlip(p=0.1),
                    A.VerticalFlip(p=0.1),
                    A.Blur(blur_limit=3, p=0.2),
                    A.CLAHE(p=0.15),
                    A.RandomRotate90(p=0.1),
                    #A.ElasticTransform(p=0.05, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    #A.GridDistortion(p=0.15),
            ]
            self.resize = A.Compose(T_resize, bbox_params=A.BboxParams(format='pascal_voc', label_fields=["labels"]))
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format='pascal_voc', label_fields=["labels"]))
        
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            print(e)
    
    
    def __call__(self, im: np.ndarray, target: object, p=1.0) -> np.ndarray:
        """
        Apply choosen augmentations to data

            Arguments:
                im (np.ndarray) - image data in form of numpy array (H, W, C)
                target (Dict[str, np.ndarray]) - the set of additional data - boxes, labels, masks in form of dictionary

            Returns:
                the augmented image and the set of augmented additional data - boxes, labels, masks in form of dictionary, assembeled in one dictionary
        """
        new = self.resize(image=im, masks=target["masks"], bboxes=target["boxes"], labels=target["labels"])
        if self.transform and self.enable:
            new = self.transform(image=new["image"], masks=new["masks"], bboxes=new["bboxes"], labels=new["labels"]) 
        return new
    
@torch.no_grad()
def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
    grcnn = torchvision.models.detection.transform.GeneralizedRCNNTransform(min_size=500, max_size=1333, image_mean=mean, image_std=std)
    img = grcnn(x)[0].tensors
    return img


@torch.no_grad()
def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    # Denormalize RGB images x per ImageNet stats in BCHW format, i.e. = x * std + mean
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x

def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def hist_equalize(im, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def hist_equalize(im, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB
