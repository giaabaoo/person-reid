import argparse
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from lib.config import cfg
from lib.engine.inference import inference_one_sample
from lib.models.model import build_model
from lib.data import make_data_loader

from lib.utils.checkpoint import Checkpointer
from lib.utils.comm import get_rank, synchronize
from lib.utils.directory import makedir
from lib.utils.logger import setup_logger
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
import json
import pdb
from nltk.tokenize import word_tokenize


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Image-Text Matching Inference"
    )
    parser.add_argument(
        "--root",
        default="./",
        help="root path",
        type=str,
    )
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--checkpoint-file",
        default="",
        metavar="FILE",
        help="path to checkpoint file",
        type=str,
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--load-result",
        help="Use saved reslut as prediction",
        action="store_true",
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.ROOT = args.root
    cfg.freeze()

    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = os.path.join(
        args.root, "./output", "/".join(args.config_file.split("/")[-2:])[:-5]
    )
    
    checkpointer = Checkpointer(model, save_dir=output_dir)
    _ = checkpointer.load(args.checkpoint_file)
    
    ### Load one hot json
    with open("../inference_utils/onehot_test.json", "r") as f:
        onehot_dict = json.load(f)
    
    image_path = "./datasets/CUHK-PEDES/imgs/train_query/p8848_s17661.jpg"
    image = Image.open(image_path).convert("RGB")
    
    all_image_path = "./datasets/CUHK-PEDES/imgs"
    descriptions = "The"
    # onehot = [16, 24, 6, 7, 1, 52, 9, 10, 93, 617, 22, 9, 38, 15, 31, 18, 232, 1811, 177, 29, 490]
    words_list = [word for word in word_tokenize(descriptions) if word.isalpha()]    
    onehot = [onehot_dict[i.lower()] for i in words_list]

    # output_folder = os.path.join(output_dir, "inference_records", image_path.split("/")[-1].replace(".jpg", ""))
    # makedir(output_folder) 
    
    output_folders = list()
    dataset_names = cfg.DATASETS.TEST
    for dataset_name in dataset_names:
        output_folder = os.path.join(output_dir, "inference_records", dataset_name)
        makedir(output_folder)
        output_folders.append(output_folder)
    
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    
    for output_folder, dataset_name, data_loader_val in zip(
        output_folders, dataset_names, data_loaders_val
    ):
        logger = setup_logger("PersonSearch", output_folder, get_rank())
        logger.info("Using {} GPUs".format(num_gpus))
        logger.info(cfg)
        

        pred_image_ids = inference_one_sample(
                    model,
                    data_loader_val,
                    (image, descriptions, onehot),
                    device=cfg.MODEL.DEVICE,
                    output_folder=output_folder,
                    save_data=True,
                    rerank=True
                )
        
    ### Retrieve results from the database
    with open("/home/dhgbao/CV/Person_ReID/TextReID/inference_utils/id_to_path.json", "r") as f:
        id_to_path = json.load(f)
        
    image = Image.open(image_path)
    image.save(os.path.join(output_folder, "query.jpg"))
    
    for idx, pred in enumerate(pred_image_ids):
        image = Image.open(os.path.join(all_image_path, id_to_path[str(pred.item())]))
        image.save(os.path.join(output_folder, "rank_" + str(idx) + ".jpg"))
        print("Saving ", os.path.join(output_folder, "rank_" + str(idx) + ".jpg"), "...")

if __name__ == "__main__":
    main()
