import torch
from torch import nn

from typing import List, Dict, Tuple, Union, Optional

import torchmetrics 
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pytorch_lightning as pl


from model.base_model import build_model
import tqdm

import numpy as np
from typing import List, Dict, Tuple



def list2dict(target_batch: List[List[torch.tensor]]) -> List[Dict[str, torch.tensor]]:
    """
    Transform batch data from the form of sequences of lists to the sequences of dicts
        Arguments:
            target_batch - additional data (bboxes, masks, labels, etc.) in form of sequences of lists
        Returns:
            additional data (bboxes, masks, labels, etc.) in form of sequences of dicts
    """
    targets_list = []
    for b in target_batch:
        image_id, masks, boxes, labels, area, iscrowd = b
        targets = {}
        targets["masks"] = masks
        targets["boxes"] = boxes
        targets["labels"] = labels
        targets["area"] = area
        targets["iscrowd"] = iscrowd
        targets["image_id"] = image_id
        targets_list.append(targets)
    return targets_list


class LightningMRCNN(pl.LightningModule):
        """
        Lightning module of Mask RCNN
            Attributes:
                hyperparameters (Dict[str, float]) - base model Mask RCNN hyperparameters
                model_name (str) - backbone name
                model (nn.Module) - main model
                val_evaluator (nn.Module) - evaluation metrics calculation module
                last_learning_rate (float) - init learning rate value
        """
    
    def __init__(self, name: str, hyperparameters: object, device:torch.device) -> None:
        
        """
        Lightning module of Mask RCNN constructor
            Arguments:
                name (str) - backbone name
                hyperparameters (Dict[str, float]) - base model Mask RCNN hyperparameters
                device (nn.torch.device) - GPU/CPU/TPU device
        """
        super().__init__()
        self.hyperparameters = hyperparameters
        self.model_name = name
        
        in_channels = hyperparameters["in_channels"]
        pyramid_channels = hyperparameters["pyramid_channels"]
        depth = hyperparameters["depth"]
        num_classes = hyperparameters["num_classes"]
        
        self.model = build_model(
            name,
            in_channels=in_channels, 
            pyramid_channels=pyramid_channels, 
            depth=depth, 
            num_classes=num_classes,
        ).to(device)
        self.to(device)
        
        self.val_evaluator = MeanAveragePrecision(box_format="xyxy").to(device)
        self.last_learning_rate = hyperparameters["lr0"]
        
    def forward(self, x: List[torch.tensor], targets: List[Dict[str, torch.tensor]]) -> torch.tensor:
        """
        Model forward pass
            Arguments:
                x (List[torch.tensor]) - imges part of the batch
                targets (List[Dict[str, torch.tensor]]) - addirional part of the batch (masks, labels, bboxes, etc.)

            Returns:
                output value (Union[List[Dict[str, torch.tensor]], Dict[str, torch.tensor]]) - loss if there is a training mode, and prediction otherwise
        """
        x = [i.to(self.device) for i in x]
        if self.training:
            targets = [{k:v.to(self.device) for k, v in i.items()} for i in targets]
            out = self.model.forward(x, targets)
        else:
            out = self.model.forward(x)
            
        del x, targets
        return out
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.tensor:
        self.train()
        tensors, targets = batch
        targets = list2dict(targets)
        pred = self.forward(tensors, targets)
        loss = sum(loss for loss in pred.values())
        #loss = torch.stack([i for i in pred.values()]).mean()
        loss_info = {k: v.item() for k, v in pred.items()}
        
        self.log("step_loss", loss)
        return {"loss": loss, "loss_info": loss_info}
    
    def training_epoch_end(self, training_step_outputs):
        train_loss = float(
            torch.stack([o["loss"] for o in training_step_outputs]).mean().cpu().numpy()
        )
        loss_classifier = float(np.mean([o["loss_info"]["loss_classifier"] for o in training_step_outputs]))
        
        loss_classifier = float(np.mean([o["loss_info"]["loss_classifier"] for o in training_step_outputs]))
        loss_box_reg = float(np.mean([o["loss_info"]["loss_box_reg"] for o in training_step_outputs]))
        loss_mask = float(np.mean([o["loss_info"]["loss_mask"] for o in training_step_outputs]))
        loss_objectness = float(np.mean([o["loss_info"]["loss_objectness"] for o in training_step_outputs]))
        loss_rpn_box_reg = float(np.mean([o["loss_info"]["loss_rpn_box_reg"] for o in training_step_outputs]))
        
        self.log("train_loss", train_loss)
        
        self.log("loss_classifier", loss_classifier)
        self.log("loss_box_reg", loss_box_reg)
        self.log("loss_mask", loss_mask)
        self.log("loss_objectness", loss_objectness)
        self.log("loss_rpn_box_reg", loss_rpn_box_reg)
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        tensors, targets = batch
        targets = list2dict(targets)
        
        pred = self.forward(tensors, targets)
        
        self.val_evaluator.update(pred, targets)

    
    def validation_epoch_end(self, validation_step_outputs):

        metrics = self.val_evaluator.compute()
        """
        #map
        map_ = float(
             torch.stack([o["map"] for o in validation_step_outputs]).mean().cpu().numpy()
         )
        self.log("map", map_, on_epoch=True)
        
        #map_50
        map_50 = float(
             np.mean([o["map_50"] for o in validation_step_outputs])
         )
        self.log("map_50", map_50, on_epoch=True)
        
        #mar_1
        mar_1 = float(
             np.mean([o["mar_1"] for o in validation_step_outputs])
         )
        self.log("mar_1", mar_1, on_epoch=True)
        
        #mar_100
        mar_100 = float(
             np.mean([o["mar_100"] for o in validation_step_outputs])
        )
        self.log("mar_100", mar_100, on_epoch=True)
        """

        map_ = metrics["map"].item()
        self.log("map", map_, on_epoch=True)

        map_50 = metrics["map_50"].item()
        self.log("map_50", map_50, on_epoch=True)

        mar_100 = metrics["mar_100"].item()
        self.log("mar_100", mar_100, on_epoch=True)

        mar_1 = metrics["mar_1"].item()
        self.log("mar_1", map_, on_epoch=True)
        
        del self.val_evaluator
        self.val_evaluator = MeanAveragePrecision(box_format="xyxy").to(self.device)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.last_learning_rate, momentum=0.9)
        sch = lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=16, gamma=0.9)
        return {
            "optimizer":optimizer,
            "lr_scheduler" : {"scheduler" : sch, "monitor" : "step_loss",}
        }
    
