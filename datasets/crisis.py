import torch
import torch.nn as nn
from PIL import Image
import pickle
import json
import os
import csv
torch.random.manual_seed(42)

class CRISIS(nn.Module):
    label_to_id = {'affected_individuals':0,
                    'infrastructure_and_utility_damage':1,
                    'injured_or_dead_people':2,
                    'missing_or_found_people':3,
                    'rescue_volunteering_or_donation_effort':4,
                    'vehicle_damage':5,
                    'other_relevant_information':6,
                    'not_humanitarian':7}
    def __init__(self, split, dropout_p=None, datafile = '/home1/dyang165/Datasets/Crisis-MMD/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_train.tsv', datadir = '/scratch1/dyang165/Datasets/Crisis-MMD/CrisisMMD_v2.0'):
        if split != 'train':
            datafile = datafile.replace('train',split)
        categories, data = self.read_tsv(datafile)
        for row in data:
            row[4] = os.path.join(datadir, row[4])
        self.data = data
        self.num_classes = len(self.label_to_id) 

    def read_tsv(self, fname):
        data = []
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter="\t", quotechar='"')
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
        label = item[5]
        text = item[3]
        imagefile = item[4]

        # Get inputs
        image = Image.open(imagefile).convert("RGB")

        # get labels
        label = self.label_to_id[label]
        return {'text':text, 'image':image, 'labels':label}

    
if __name__ == '__main__':
    db = CRISIS("dev")
    print(db.__getitem__(0)) 
