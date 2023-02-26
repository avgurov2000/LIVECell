import os, sys
import argparse
from pycocotools.coco import COCO
import tqdm
import json
from collections import defaultdict
import re
import numpy as np
import shutil


annotations = {
    "0": {
        "name": "A172",
        "path": r"https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/a172/"
    },
    "1": {
        "name": "BT474",
        "path": r"https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/bt474/",
    },
    "2": {
        "name": "BV-2",
        "path": r"https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/bv2/",
    },
    "3": {
        "name": "Huh7",
        "path": r"https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/huh7/",
    },
    "4": {
        "name": "MCF7",
        "path": r"https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/mcf7/",
    },
    "5": {
        "name": "SH-SHY5Y",
        "path": r"https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/shsy5y/",
    },
    "6": {
        "name": "SkBr3",
        "path": r"https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/skbr3/",
    },
    "7": {
        "name": "SK-OV-3",
        "path": r"https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell_single_cells/skov3/",
    },
    "8": {
        "name": "LIVECell",
        "path": r"http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/annotations/LIVECell/livecell_coco_",
    }
}

def plot_stage(stage: str):
    stage_len = len(stage)
    full_row_len = 6 * stage_len
    mid_row_len = full_row_len - stage_len - 2
    
    lrow = mid_row_len//2
    rrow = mid_row_len - lrow
    
    print("#" * full_row_len)
    print("#" * lrow + " " + stage + " " + "#"* rrow)
    print("#" * full_row_len)
    print()
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default="data/coco/annotations", help='path to save downloaded data') 
    parser.add_argument('--data_path', type=str, default="data/coco/images", help='path to save downloaded data')
    parser.add_argument('--ratio', type=float, default=0.15, help='test/train ratio')
    return parser.parse_args()

def annotations_download():
    
    print("Choose dataset for download:")
    for k, v in annotations.items():
        name = v["name"]
        print(f"Input {k} for {name} dataset")
    input_value = input()
    
    chosen_path = annotations[input_value]["path"]
    annotations_paths = {
        "train.json": chosen_path+"train.json",
        "val.json": chosen_path+"val.json",
        "test.json": chosen_path+"test.json",
    }
    
    plot_stage("Annotations download")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    for k, v in annotations_paths.items():
        command = f"wget -O {os.path.join(save_path, k)} {v}"
        os.system(command)
        

def annotations_proces(im_path, ann_path):
    
    plot_stage("Annotations processing")
    
    imgs_cnt = 0
    anns_cnt = 0
    skip_cnt = 0
        
    image_names = os.listdir(im_path)
    
    for k in ["train.json", "val.json", "test.json"]:
        curr_annotation_path = os.path.join(ann_path, k)
        coco=COCO(curr_annotation_path)
        
        img_ids = coco.getImgIds()
        img_json = coco.loadImgs(img_ids)
        
        for img in tqdm.tqdm(img_json):
            
            if img["file_name"] not in image_names:
                skip_cnt += 1
                continue
        
            imgs_cnt += 1
            ann_ids = coco.getAnnIds(imgIds=img['id'])
            ann_json = coco.loadAnns(ann_ids)
            anns_cnt += len(ann_json)

            image_label = dict()
            image_label["img_info"] = img
            image_label["annotations"] = ann_json
            
            ann_save_name = f'{image_label["img_info"]["id"]}_{image_label["img_info"]["file_name"]}'
            ann_save_name = os.path.splitext(ann_save_name)[0] + ".json"
            ann_save_name = os.path.join(ann_path, ann_save_name)
            
            with open(ann_save_name, 'w') as fjson:
                json.dump(image_label, fjson)
            
        del coco
        command = f"rm {curr_annotation_path}"
        os.system(command)
    print(f"Processed images: {imgs_cnt}, processed annotations: {anns_cnt}, skiped images: {skip_cnt}")
    

def annotations_split(im_path, ann_path):
    
    plot_stage("Annotations split")
    
    ann_files = os.listdir(ann_path)
    img_files = os.listdir(im_path)
    img_names = [os.path.splitext(i)[0] for i in img_files]
    assert len(img_files) == len(img_names), "Not correct length"
    
    img2ann = defaultdict(lambda: set())

    for ann in tqdm.tqdm(ann_files):
        ann_id, ann_img_name = re.findall(r"^\d+", ann), os.path.splitext(re.sub(r"^\d+_", "", ann))[0]
        assert len(ann_id) and ann_img_name in img_names

        img2ann[ann_img_name].add(ann)
        
    img_nonunique = [k for k, v in img2ann.items() if len(list(v))>1]
    print(f"Non unique data: {len(img_nonunique)}")
    
    train, test = [], []

    for img, ann_set in tqdm.tqdm(img2ann.items()):
        prob = np.random.uniform()
        for ann in list(ann_set):
            if prob < ratio:
                test.append(ann)
            else:
                train.append(ann)
                
    print(f"Train set length: {len(train)}")
    print(f"Test set length: {len(test)}")
    print(f"Ratio of test/train: {len(test)/len(train)}.")
    
    os.makedirs(os.path.join(ann_path, "train"))
    os.makedirs(os.path.join(ann_path, "test"))
    
    for f in train:
        src_file = os.path.join(ann_path, f)
        dst_file = os.path.join(os.path.join(ann_path, "train"), f)
        shutil.move(src_file, dst_file)
    
    for f in test:
        src_file = os.path.join(ann_path, f)
        dst_file = os.path.join(os.path.join(ann_path, "test"), f)
        shutil.move(src_file, dst_file)
    

        
if __name__ == "__main__":
    
    opt = parse_opt()
    save_path = opt.save_path
    data_path = opt.data_path
    ratio = opt.ratio
    
    assert os.path.exists(data_path), f"data path '{data_path}' does not exist."
    assert ratio > 0 and ratio <= 1, f"test/train ratio must be in interval (0, 1], but got {ratio}."
    
    try:
        annotations_download()
    except Exception as e:
        raise e
        
    try:
        annotations_proces(data_path, save_path)
    except Exception as e:
        raise e
        
    try:
        annotations_split(data_path, save_path)
    except Exception as e:
        raise e
        
    
    
    
