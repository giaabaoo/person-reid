import os

import torch.utils.data
import torch.utils.data.distributed

from lib.config import cfg
from lib.engine.inference import inference_one_sample
from lib.models.model import build_model
from lib.data import make_data_loader

from lib.utils.checkpoint import Checkpointer
from lib.utils.comm import get_rank
from lib.utils.directory import makedir
from lib.utils.logger import setup_logger
from PIL import Image
import json
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

def predict_results(image, text):
    config_file = "./configs/cuhkpedes/moco_gru_cliprn50_ls_bs128_2048.yaml"
    
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    cfg.merge_from_file(config_file)
    cfg.merge_from_list([])
    cfg.ROOT = "./"
    cfg.freeze()

    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = os.path.join(
        cfg.ROOT, "./output", "/".join(config_file.split("/")[-2:])[:-5]
    )
    checkpoint_file = "./pretrained/moco_gru_cliprn50_ls_bs128_2048.pth"
    checkpointer = Checkpointer(model, save_dir=output_dir)
    _ = checkpointer.load(checkpoint_file)
    
    ### Load one hot json
    with open("./inference_utils/onehot_test.json", "r") as f:
        onehot_dict = json.load(f)
    
    all_image_path = "./datasets/CUHK-PEDES/imgs"
    words_list = [word for word in word_tokenize(text) if word.isalpha()]    
    onehot = [onehot_dict[i.lower()] for i in words_list]
    
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
                    (image, text, onehot),
                    device=cfg.MODEL.DEVICE,
                    output_folder=output_folder,
                    save_data=True,
                    rerank=True
                )
        
    ### Retrieve results from the database
    with open("./inference_utils/id_to_path.json", "r") as f:
        id_to_path = json.load(f)
    
    images_list = []
    for pred in pred_image_ids:
        images_list.append(Image.open(os.path.join(all_image_path, id_to_path[str(pred.item())])))
    return images_list

if __name__ == "__main__":
    predict_results()
