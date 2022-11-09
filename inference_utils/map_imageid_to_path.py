import json

if __name__ == "__main__":
    path = "/home/dhgbao/CV/Person_ReID/datasets/Text-based/CUHK-PEDES/annotations/test.json"
    
    with open(path, "r") as f:
        data = json.load(f)

    id_to_path = {}
    for sample in data['annotations']:
        id_to_path[sample['image_id']] = sample['file_path']
    
    with open("id_to_path.json", "w") as f:
        json.dump(id_to_path, f)