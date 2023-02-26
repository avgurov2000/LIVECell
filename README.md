# LIVECell
LIVECell dataset instance segmentation

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Loading data](#loading)
* [Training](#training)
* [Inference](#inference)
* [Results](#results)

## General info

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



	
## Technologies
Project is created with:
* [Streamlit](https://streamlit.io/) version: 1.18.1
* [Torch](https://pytorch.org/) version: 1.13.1
* [Torchvision](https://pytorch.org/vision/stable/index.html) version: 0.14.1
* [Open CV](https://opencv.org/) version: 4.7.0
* [albumentations](https://albumentations.ai/) version: 1.3.0

## Loading data
Run data load example (save_path - path to storing downloaded images):
```
$ python3 load_data.py --save_path=./data/coco/images
```
Run annotations load + split example (save_path - path to storing downloaded data, data_path - path to stored inmages, ratio - test/train ratio):
```
$ python3 load_annotation.py --save_path=./data/coco/a0 --data_path=./data/coco/images --ratio=0.2
```
## Training
Run training script example:
```
$ python3 train.py --model=efficientnet-b0 --config=./configs/config_final.yaml /
		   --batch_size=2 --epoch=25 --device=cuda --seed=1488 --num_workers=4
```

## Inference
for inference you should have .onnx and .ckpt data in models_arh folder with certain names (it is kind of hard code implementation, but it supposed to be an inference of already trained and prepared model =|, but these files too large forr being uploaded)
```
$ streamlit run ./app/main.py
```
## Results
Why is it needed to have .cpkt file besides .onnx model? So, I recently noticed that the .onnx model does not display masks, more precisely, it displays them as absolute zeros, while the model loaded from .ckpt files displays them (not entirely true, often the mask is equal to the whole space inside the bbox, there is an idea how to fix this, but hands did not reach). 
For this reason, if you deploy the .onnx model, then it will draw a bbox, sad, of course, but I didn’t come up with anything else.

Below are the results of the model:

### initial image:
<img
  src="/pics/init_img.png"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 300px">


### .onnx model output:
<img
  src="/pics/onnx_img.png"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 300px">

### .ckpt model output:
<img
  src="/pics/ckpt_img.png"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 300px">
