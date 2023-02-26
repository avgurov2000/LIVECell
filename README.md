# LIVECell
LIVECell dataset instance segmentation

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Training](#training)

## General info
This project is simple Lorem ipsum dolor generator.
	
## Technologies
Project is created with:
* [Streamlit](https://streamlit.io/) version: 1.18.1
* [Torch](https://pytorch.org/) version: 1.13.1
* [Torchvision](https://pytorch.org/vision/stable/index.html) version: 0.14.1
* [Open CV](https://opencv.org/) version: 4.7.0
* [albumentations](https://albumentations.ai/) version: 1.3.0
	
## Training
To run this project, install it locally using npm:

```
$ python3 train.py --model=efficientnet-b0 --config=./configs/config_final.yaml --batch_size=2 --epoch=25 --device=cuda --seed=1488 --num_workers=4
```

#### This repo comprises following folder:
- app - tools for deployed model 
- configs - model training configs
- dataset - tools for data processing and loading
- model - main model source code
- models_arch - .onnx and .ckpt models storing
- utils - utils and help tools

#### And folowwing files:
- load_annotation.py - load annotations
- load_data.py - load data (all images!!!)
- onnx_create.py - create .onnx model from .ckpt
- train.py - train model


