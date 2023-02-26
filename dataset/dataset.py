import os
import re
import json
import numpy as np
from .augmentations import Albumentations
from .mask import CocoSegmentation

import cv2

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple, Union, Optional, Callable, Any


def xywh2xyxy(bbox: List[float]) -> List[float]:
    x_min, y_min, width, height = bbox
    x_min, y_min, x_max, y_max = x_min, y_min, x_min+width, y_min+height
    return [x_min, y_min, x_max, y_max]
    
class CellDataset(Dataset):
    
    def __init__(
        self,
        path: str,
        data_path: str,
        augmentations: bool = False,
    ) -> None:
        
        self.path = path
        self.data_path = data_path
        self.augmentations = Albumentations(enable=True) if augmentations else Albumentations(enable=False)
        
        self.annotation_path = path
        self.image_path = data_path
        self.keys = ["image_id", "masks", "boxes", "labels", "area", "iscrowd"]

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Path '{self.path}' does not exist.")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path '{self.data_path}' does not exist.")
        if not os.path.exists(self.annotation_path):
            raise FileNotFoundError(f"Annotation path '{self.annotation_path}' does not exist.")
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image path '{self.image_path}' does not exist.")
            
            
        files = os.listdir(path)
        self.files = [os.path.join(self.annotation_path, re.sub(r"\n", "", i)) for i in files]
        if not all([os.path.exists(i) for i in self.files]):
            raise FileNotFoundError(f"Not all annotations files are presented in '{self.annotation_path}'")
        
    def _get_row(self, idx):
        with open(self.files[idx], "r") as fp:
            row_data = json.load(fp)
        return row_data
    
    @torch.no_grad()
    def __getitem__(self, idx):
        row_data = self._get_row(idx)
        img_json, ann_json = row_data["img_info"], row_data["annotations"]
        
        height, width = img_json["height"], img_json["width"]
        image_file_name = os.path.join(self.image_path, img_json["file_name"])
        im = cv2.imread(image_file_name, cv2.IMREAD_UNCHANGED)
        
        target = {}
        masks = [CocoSegmentation.annToMask((height, width), ann) for ann in ann_json]
        boxes = [xywh2xyxy(ann["bbox"]) for ann in ann_json]
        labels = [ann["category_id"] for ann in ann_json]
        area = [ann["area"] for ann in ann_json]
        iscrowd = [ann["iscrowd"] for ann in ann_json]
        image_id = img_json["id"]
        
        target["masks"] = masks
        target["boxes"] = boxes
        target["labels"] = labels
        
        new = self.augmentations(im, target)
        im = new["image"]
        target["masks"] = new["masks"]
        target["boxes"] = new["bboxes"]
        target["labels"] = new["labels"]
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = image_id
        
        im = np.expand_dims(im, 0)
        im = np.ascontiguousarray(im)
        im = torch.as_tensor(im).float()/255
        
        target["image_id"] = torch.as_tensor(target["image_id"], dtype=torch.int)
        target["masks"] = torch.as_tensor(np.expand_dims(np.stack(target["masks"]), 1), dtype=torch.uint8)
        target["boxes"] = torch.as_tensor(np.stack(target["boxes"]), dtype=torch.float)
        target["labels"] = torch.as_tensor(np.stack(target["labels"]), dtype=torch.int64)
        target["area"] = torch.as_tensor(np.stack(target["area"]), dtype=torch.float)
        target["iscrowd"] = torch.as_tensor(np.stack(target["iscrowd"]))
        
        return im, [target[key] for key in self.keys]
    
    def __len__(self):
        return len(self.files)
    
