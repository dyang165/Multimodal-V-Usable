import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import pickle
import json
import os
import numpy as np
torch.random.manual_seed(42)

class MMIMDB(Dataset):
    def __init__(self, split, train_p = 1, dropout_p=None, datadir = '/home1/dyang165/Datasets/mmimdb/mmimdb/', label_map = '/home1/dyang165/Datasets/mmimdb/mmimdb_genre_label_map_23_classes.pkl'):
        self.datadir = datadir
        with open(os.path.join(datadir, 'split.json'), 'r') as f:
            self.ids = json.load(f)[split]
        
        with open(label_map, 'rb') as f:
            self.labels_to_ids = pickle.load(f)
        self.num_classes = len(self.labels_to_ids)
        
        if split == 'dev':
            self.ids = self.ids
       
        if train_p == None:
            train_p = 1 
        print(f"{split}_p:",train_p)
        self.index_array = np.arange(len(self.ids))
        if train_p != None and train_p != 1:
            np.random.shuffle(self.index_array) 
            self.index_array = self.index_array[:int(train_p*len(self.ids))]
 
    def __len__(self):
        return len(self.index_array)

    def __getitem__(self, x):
        x = self.index_array[x]
        itemid =  self.ids[x]
        imagefile = os.path.join(self.datadir,f'data/{itemid}.jpeg')
        labelfile = os.path.join(self.datadir,f'data/{itemid}.json')

        # Get inputs
        image = Image.open(imagefile).convert("RGB")
        with open(labelfile, 'r') as f:
            data = json.load(f)

        text = " ".join(data['plot'])
        
        # get labels
        labels = [self.labels_to_ids[item] for item in data['genres'] if item in self.labels_to_ids]
        return {'text':text, 'image':image, 'labels':labels}

    
if __name__ == '__main__':
    db = MMIMDB("train", train_p=0.01)
    print(db.labels_to_ids)
    print(db.__getitem__(0))
