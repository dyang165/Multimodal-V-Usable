import torch
import torch.nn as nn
from PIL import Image
import pickle
import json
import os
torch.random.manual_seed(42)

class HATEFUL(nn.Module):
    def __init__(self, split, dropout_p=None, datadir = '/home1/dyang165/Datasets/Hateful-Memes/data', label_map = '/home1/dyang165/Datasets/Hateful-Memes/hateful_memes_label_map.pkl'):
        self.datadir = datadir
        self.data = self.load_jsonl(os.path.join(datadir, f'{split}.jsonl'))
        
        with open(label_map, 'rb') as f:
            self.labels_to_ids = pickle.load(f)
        self.num_classes = len(self.labels_to_ids)

    def load_jsonl(self, fname):
        with open(fname, 'r') as json_file:
            json_list = list(json_file)
        data = []
        for json_str in json_list:
            result = json.loads(json_str)
            data.append(result)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, x):
        item = self.data[x]
        imagefile = os.path.join(self.datadir,item['img'])
        label = item['label']
        text = item['text']

        # Get inputs
        image = Image.open(imagefile).convert("RGB")
        return {'text':text, 'image':image, 'labels':label}

    
if __name__ == '__main__':
    db = HATEFUL("train")
    print(len(db))
    for i in range(6500,6510):
        print(db.__getitem__(i))
