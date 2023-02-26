import collections


import torch
from torch import nn

import torchvision
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator

import segmentation_models_pytorch as smp
from tqdm import tqdm as tqdm

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation

encoder_names = sorted(smp.encoders.get_encoder_names())

class Encoder(nn.Module):
    
    def __init__(
        self, 
        body: nn.Module, 
        fpn: nn.Module,
        out_channels: int,
    ) -> None:
        
        super().__init__()
        self.body = body
        self.fpn = fpn
        self.out_channels = out_channels
        
    def forward(self, x) -> torch.tensor:
        features_out = self.body(x)[-4:] #features_out = self.body(x)[-4:]
        features_out = collections.OrderedDict( (str(k), v) for k, v in enumerate(features_out))
        fpn_out = self.fpn(features_out)
        return fpn_out
        
        
        
        
def get_encoder(
    encoder_name: str, 
    in_channels: int = 3, 
    pyramid_channels: int = 256,
    depth: int = 5,
    weights="imagenet",
):
    if encoder_name not in encoder_names:
        raise ValueError(f"There is no model with name '{encoder_name}'. Available models are: {encoder_names}.")
        
    body = smp.encoders.get_encoder(name=encoder_name, in_channels=in_channels, depth=depth, weights="imagenet")
    
    out_channels = body.out_channels
    #fpn = FeaturePyramidNetwork(out_channels[-4:], pyramid_channels, extra_blocks=LastLevelMaxPool())
    fpn = FeaturePyramidNetwork(out_channels[-4:], pyramid_channels)
    
    model = Encoder(body=body, fpn=fpn, out_channels=pyramid_channels)
    
    return model


def build_model(
    encoder_name: nn.Module, 
    in_channels: int = 1,
    pyramid_channels: int = 256,
    depth: int = 5,
    num_classes: int = 2,
) -> None:
    
    min_size=260 #520
    max_size=352 #704
    box_detections_per_img = 512 #512
    anchor_generator = AnchorGenerator(
    sizes=((4,), (9,), (17,), (31,),),  #sizes=((4,), (9,), (17,), (31,), (64,),), 
    aspect_ratios=((0.5, 1.0, 1,5, 2.0))
    )
    #roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
    #mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=14, sampling_ratio=2)
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)

        
    encoder = get_encoder(encoder_name, in_channels=in_channels, pyramid_channels=pyramid_channels, depth=depth)
    model = MaskRCNN(
        encoder,
        min_size=min_size,
        max_size=max_size,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        box_detections_per_img=box_detections_per_img,
    )
    
    
    #grcnn = torchvision.models.detection.transform.GeneralizedRCNNTransform(min_size=260, max_size=1333, image_mean=IMAGENET_MEAN, image_std=IMAGENET_STD)
    #model.transform = grcnn
    
    for i in model.backbone.body.parameters():
        i.requires_grad = True
        
    return model
