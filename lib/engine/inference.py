import datetime
import logging
import os
import time
from collections import defaultdict

import torch
from tqdm import tqdm

from lib.data.metrics import evaluation
from lib.data.metrics import retrieve_results
from lib.utils.comm import all_gather, is_main_process, synchronize
from torchvision import transforms
from lib.utils.caption import Caption


import pdb
from PIL import Image


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = defaultdict(list)
    for batch in tqdm(data_loader):
        images, captions, image_ids = batch
        images = images.to(device)
        captions = [caption.to(device) for caption in captions]
        
        with torch.no_grad():
            output = model(images, captions)
        for result in output:
            for img_id, pred in zip(image_ids, result):
                results_dict[img_id].append(pred)
    return results_dict

def compute_on_sample(model, sample, device):
    model.eval()
    results = list()
    image, caption, onehot = sample
    #transform compose resize and totensor
    tf = transforms.Compose([
        transforms.Resize((384, 128)),
        transforms.ToTensor()])
    # BAO: add transform to resize image
    image = tf(image).to(device).unsqueeze(0)
    # one hot encode the caption kh xai onehot
    # caption = Caption(caption)
    caption = torch.tensor(onehot)
    caption = Caption([caption], max_length=105, padded=False).to(device)
    # caption.add_field("img_path", img_path)
        
    with torch.no_grad():
        output = model(image, caption)
    for result in output:
        results.append(result)
            
    return results

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("PersonSearch.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )
    return predictions


def inference(
    model,
    data_loader,
    dataset_name="cuhkpedes-test",
    device="cuda",
    output_folder="",
    save_data=True,
    rerank=True,
):
    logger = logging.getLogger("PersonSearch.inference")
    dataset = data_loader.dataset
    logger.info(
        "Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset))
    )

    predictions = None
    if not os.path.exists(os.path.join(output_folder, "inference_data.npz")):
        # convert to a torch.device for efficiency
        device = torch.device(device)
        num_devices = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        start_time = time.time()

        predictions = compute_on_dataset(model, data_loader, device)
        # wait for all processes to complete before measuring the time
        synchronize()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        logger.info(
            "Total inference time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(dataset), num_devices
            )
        )
        predictions = _accumulate_predictions_from_multiple_gpus(predictions)

        if not is_main_process():
            return

    return evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        save_data=save_data,
        rerank=rerank,
        topk=[1, 5, 10],
    )

def inference_one_sample(
    model,
    data_loader,
    sample,
    device="cuda",
    output_folder="",
    save_data=True,
    rerank=True
):
    image, description, onehot = sample
    logger = logging.getLogger("PersonSearch.inference")
    dataset = data_loader.dataset
    logger.info(
        "Starting inference on query image."
    )
    
    predictions = None
    
    device = torch.device(device)
   
    start_time = time.time()
    inference_sample = image, description, onehot
    sample_prediction = compute_on_sample(model, inference_sample, device)
    
    if not os.path.isfile(os.path.join(output_folder, "database_features.npz")):
        predictions = compute_on_dataset(model, data_loader, device)
        predictions = _accumulate_predictions_from_multiple_gpus(predictions)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    print("Total inference time: ", total_time_str)

    if not is_main_process():
        return

    return retrieve_results(
        dataset=dataset,
        predictions=predictions,
        sample_prediction=sample_prediction,
        output_folder=output_folder,
        save_data=save_data,
        rerank=rerank,
        topk=[1, 5, 10],
    )