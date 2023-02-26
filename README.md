# LIVECell
LIVECell dataset instance segmentation

#### This repo comprises following folder:
- app -- tools for deployed model 
- configs -- model training configs
- dataset -- tools for data processing and loading
- model --main model source code
- models_arch -- .onnx and .ckpt models storing
- utils -- utils and help tools

#### And folowwing files:
- load_annotation.py -- load annotations
- load_data.py -- load data (all images!!!)
- onnx_create.py -- create .onnx model from .ckpt
- train.py -- train model
