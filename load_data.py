import os, sys
import logging
import argparse

data_path = r"http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip"

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default="data/coco", help='path to save downloaded data') 
    return parser.parse_args()

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


if __name__ == "__main__":

    is_wrutable = os.access('./', os.R_OK & os.W_OK & os.X_OK) # Check if we can write and read from current dir and execute files
    if not is_wrutable:
        raise PermissionError(f"You do not have enough permissions for directory {os.path.dirname(os.path.abspath(__file__))}.")
     
    logging.basicConfig(level=logging.DEBUG, filename='loadlog.log', format='%(asctime)s %(levelname)s:%(message)s')
    logging.info(f"Download data from {data_path}") ###Log
    
    opt = parse_opt()
    save_path = opt.save_path
    data_tmp_name = os.path.join(save_path, "data_tmp" + os.path.splitext(data_path)[-1])

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logging.info(f"Dir '{save_path}' created.") ###Log
    else:
        logging.info(f"Dir '{save_path}' already exists.") ###Log

    logging.info("Data download:start")
    command = f"wget -O {data_tmp_name} {data_path}"
    logging.info(f"terminat: ${command}")
    os.system(command)
    logging.info(f"Data download:done")
    
    logging.info("Data unzip:start")
    logging.info(f"terminat: ${command}")
    command = f"unzip -q {data_tmp_name} -d {save_path}; rm {data_tmp_name}"
    os.system(command)
    logging.info("Data unzip:done")
    
    logging.info("Data move:start")
    IMAGE_PATH = os.path.join(save_path, "images")
    move_path_from, move_path_to = IMAGE_PATH + f"{os.sep}**{os.sep}*.tif", IMAGE_PATH
    drop_dir_list = [os.path.join(move_path_to, i) for i in os.listdir(move_path_to)]
    drop_dir_list = [i for i in drop_dir_list if os.path.isdir(i)]
    logging.info(f"terminat: ${command}")
    command = f"mv {move_path_from} {move_path_to}; rm -d " + " ".join(drop_dir_list) 
    os.system(command)
    logging.info("Data move:done")
    
    logging.info(f"Download data finished") ###Log








