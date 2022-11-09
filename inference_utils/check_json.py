import json
from tqdm import tqdm

if __name__=="__main__":
    path = "/home/dhgbao/CV/Person_ReID/datasets/Text-based/CUHK-PEDES/annotations/train.json"
    
    with open(path,"r") as f:
        data = json.load(f)
    
    samples = []
    for sample in tqdm(data['annotations']):
        if "p8848_s17661" in sample['file_path']:
            print(sample)
        samples.append(sample['file_path'])
        
    # save samples to txt
    with open("train.txt","w") as f:
        for sample in samples:
            f.write(sample + "\n")