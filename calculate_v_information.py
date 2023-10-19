import yaml
import sys
import torch
import torch.nn as nn
import transformers
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from models.heads.heads import *
from models.heads.hydra import *
from utils.train_utils import *
from torch.distributions import Categorical
import torchmetrics
from torchmetrics import F1Score
from sklearn.metrics import f1_score
import numpy as np
import tqdm
import os
#import pdb; pdb.set_trace()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
torch.manual_seed(69)

# TODO
#DONE (1) Load eval dataset
#DONE (2) Assign uni or multimodal based on p_t, p_i
#DONE (3) Evaluate image p_t
#DONE (4) Evaluate text p_i
#DONE (5) Evaluate multimodal p_t x2
#DONE (6) Aggregate results
#(7) Make plots

def compute_v_information(model, processor, config, dataset, modality, bs=32): # Choose a large bs since we are evaluating
    mem = torch.cuda.get_device_properties(0).total_memory/1e9
    if mem > 80:
        bs = 64
    if isinstance(model, nn.DataParallel):
        model = model.module
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    

    config_task = config.copy()
    dataloader = DataLoader(dataset, batch_size = bs, collate_fn = lambda x: collate_fn(x, processor, dataset.num_classes, config_task), shuffle=True)
    total_logits = []
    total_labels = []
    save = []
    if config["task"] == 'multilabel':
        hyx = torch.zeros(dataset.num_classes)
        count = torch.zeros(dataset.num_classes)
    elif config["task"] == 'multiclass':
        hyx = 0
        count = 0
    with torch.no_grad():
        model.eval()
        it = 0
        for batch, labels in tqdm.tqdm(dataloader):
            it+=1
            if it == 50:
                break
            batch = {k:v.to(device) for k,v in batch.items()} 
            labels = labels.to(device)
            logits = model(**batch)
            if isinstance(logits, tuple):
                logits = logits[-1]
            if isinstance(logits, transformers.modeling_outputs.SequenceClassifierOutput):
                # Huggingface class
                logits = logits.logits
            ### TODO Write code for v-information ###
            if config["task"] == 'multilabel':
                probs = torch.nn.functional.sigmoid(logits)
                # See paper https://www.cs.cmu.edu/~schneide/Composite_V2.pdf
                # Take the marginal likelihood (assumes labels are independent)
                update = torch.sum(torch.log2(probs) * labels, dim = 0).cpu()
                save.append(update)
                count = count + torch.sum(labels, dim = 0).cpu()
                hyx = torch.subtract(hyx, update)
            elif config["task"] == 'multiclass':
                probs = torch.nn.functional.softmax(logits, dim=1)
                update = torch.Tensor([torch.log2(1/probs[i,x]) for i,x in enumerate(labels)])
                save.append(update)
                hyx = hyx+torch.sum(update)
                count = count + len(labels)
               # breakpoint()
            total_logits.append(logits.cpu())
            total_labels.append(labels.cpu())
            print(hyx/count)
        #save = torch.cat(save)
        #with open('results/fullmodal.pkl', 'wb') as f:
        #    pickle.dump(save, f)
        total_logits = torch.cat(total_logits).cpu().numpy()
        total_labels= torch.cat(total_labels).cpu().numpy().astype('int')
        if config["task"] == 'multilabel':
            hyx = torch.div(hyx,count)
            f1 = f1_score(total_labels, (total_logits > 0).astype('int'), average ='macro')
        elif config["task"] == 'multiclass':
            hyx = hyx / count
            f1 = f1_score(total_labels, np.argmax(total_logits, axis = 1).astype('int'), average='macro')
    return hyx, f1

if __name__ == '__main__':
    config = sys.argv[1]
    with open(config, 'r') as f:
        config = yaml.safe_load(f)
   
    dataset = get_dataset(config['dataset'], split='test')
    #dataset, _ = get_dataset(config['dataset'])

    dataset_len = len(dataset)
    print(dataset_len, dataset.num_classes)

    model, processor = get_model_and_processor(config)
    conditional_v, f1 = compute_v_information(model, processor, config, dataset, "multimodal")
    print("H(Y|X) = ", conditional_v)
    if config["task"] == 'multilabel':
        print("E[H(Y|X)] = ",torch.mean(conditional_v))
    print("macro f1 = ", f1)
