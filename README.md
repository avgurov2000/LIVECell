# LIVECell
LIVECell dataset instance segmentation

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This project is simple Lorem ipsum dolor generator.
	
## Technologies
Project is created with:
* Lorem version: 12.3
* Ipsum version: 2.33
* Ament library version: 999
	
## Setup
To run this project, install it locally using npm:

```
$ cd ../lorem
$ npm install
$ npm start
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


