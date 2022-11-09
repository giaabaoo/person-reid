import json
import os
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import pdb

if __name__ == "__main__":
    path = "/home/dhgbao/CV/Person_ReID/datasets/Text-based/CUHK-PEDES/annotations/test.json"
    
    with open(path, "r") as f:
        data = json.load(f)
    
    onehot_dict = {}
    for sample in tqdm(data['annotations']):
        words_list = word_tokenize(sample['sentence'])
        # remove special characters
        words_list = [word for word in words_list if word.isalpha()]
        
        
        for idx, word in enumerate(words_list):
            try: 
                onehot_dict[word.lower()] = sample['onehot'][idx]
            except:
                onehot_dict[word] = 0
    
    with open(os.path.join("/home/dhgbao/CV/Person_ReID/TextReID/inference_utils", "onehot_test.json"), "w") as f:
        json.dump(onehot_dict, f)
    