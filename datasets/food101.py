import torch
import torch.nn as nn
from PIL import Image
import pickle
import json
import os
import csv
import numpy as np
torch.random.manual_seed(42)

class Food101(nn.Module):
    def __init__(self, split, dropout_p=None, datafile = '/home1/dyang165/Datasets/food101/texts/train_titles.csv', datadir = '/scratch1/dyang165/Datasets/food101/images/train'):
        if split != 'train':
            datafile = datafile.replace('train',split)
            datadir = datadir.replace('train', split)
        classes = sorted(os.listdir(datadir))
        self.label_to_id = dict(list(zip(classes, np.arange(len(classes)))))
        categories, data = self.read_csv(datafile)
        self.data = data
        self.datadir = datadir
        self.num_classes = len(self.label_to_id) 

    def read_csv(self, fname):
        data = []
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter=",", quotechar='"')
            for idx, row in enumerate(reader):
                if idx == 0:
                    categories = row
                    continue
                data.append(row)
        return categories, data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, x):
        item = self.data[x]
        label = '_'.join(os.path.basename(item[0]).split('_')[:-1])
        text = item[1]
        imagefile = os.path.join(self.datadir, label, item[0])
        # Get inputs
        image = Image.open(imagefile).convert("RGB")
        # get labels
        label = self.label_to_id[label]
        return {'text':text, 'image':image, 'labels':label}

    
if __name__ == '__main__':
    db = Food101("train")
    for i in range(10):
        print(db.__getitem__(i*1000))
