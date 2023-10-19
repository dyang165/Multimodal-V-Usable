import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import pickle
import json
import os
import tqdm
torch.random.manual_seed(42)

class N24News(Dataset):
    labels = ['automobiles', 'dance', 'style', 'food', 'technology', 'travel', 'books', 'movies', 'art & design', 'opinion', 'theater', 'science', 'media', 'health', 'well', 'your money', 'education', 'economy', 'music', 'television', 'global business', 'sports', 'real estate', 'fashion & style'] 
    def __init__(self, split, dropout_p=None, datadir = '/home1/dyang165/Datasets/N24News/news', imagedir = '/home1/dyang165/Datasets/N24News/imgs/imgs'):
        self.datadir = os.path.join(datadir,f'nytimes_{split}.json')
        self.imagedir = imagedir
        with open(self.datadir, 'r') as f:
            self.data = json.load(f)
        
        self.num_classes = 24
        self.label_to_id = {}
        for idx, label in enumerate(self.labels):
            self.label_to_id[label] = idx     

    def __len__(self):
        return len(self.data)

    def __getitem__(self, x):
        image_id = self.data[x]['image_id']
        imagefile = os.path.join(self.imagedir,f'{image_id}.jpg')  
        image = Image.open(imagefile).convert("RGB") 
        text = self.data[x]['article']
        label = self.data[x]['section'].lower()
        label_id = self.label_to_id[label]
        return {'text':text, 'image':image, 'labels':label_id}

    
if __name__ == '__main__':
    db = N24News("test")
    print(db.__getitem__(0))
