import torch
import torch.nn as nn
from models.heads.heads import *
from models.heads.hydra import *
from utils.train_utils import *
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from accelerate import Accelerator
from torchmetrics import F1Score
from sklearn.metrics import f1_score
import tqdm
import numpy as np
import wandb
import yaml
import sys
import argparse
parser = argparse.ArgumentParser("Enable/Disable Wandb")
parser.add_argument('--n','--no_wandb', action='store_true')
parser.add_argument('--c','--config', action='store')
parser.add_argument('--g', '--gpu',required=False,type=int, choices=range(0,2), default=None)
args = parser.parse_args()

config = args.c
with open(config, 'r') as f:
    config = yaml.safe_load(f) 
project_name = config["project_name"]
run_name = config["run_name"] 
device_name = args.g

if not args.n:
    wandb.login()
    wandb.init(project=project_name, name = run_name)
torch.random.manual_seed(101)
device = torch.device(device_name if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

train_db, valid_db = get_dataset(config["dataset"])
num_classes = train_db.num_classes
config["num_classes"] = num_classes

print(f"There are {num_gpus} gpus")
train_dl = DataLoader(train_db, batch_size = config["bs"], collate_fn = lambda x: collate_fn(x, processor, num_classes, config), shuffle=True) 
valid_dl =  DataLoader(valid_db, batch_size = config["bs"], collate_fn = lambda x: collate_fn(x, processor, num_classes, config), shuffle=False) 

model, processor = get_model_and_processor(config)
model = nn.DataParallel(model, device_ids=[device_name])
model.to(device)

lr = config["lr"]
optim = AdamW(model.parameters(), lr=lr, eps=1e-8)
batch_per_epoch = len(train_dl)
warmup_epochs = 10
total_epochs = config["total_epochs"]
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_epochs*batch_per_epoch, num_training_steps=total_epochs*batch_per_epoch)

if config["task"] == 'multilabel':
    loss_fn = torch.nn.BCEWithLogitsLoss() #built-in sigmoid
    metric = lambda logits, labels: f1_score(labels, logits, average = 'macro')
elif config["task"] == 'multiclass':
    loss_fn = torch.nn.CrossEntropyLoss() #built-in softmax
    metric = lambda logits, labels: f1_score(labels, logits, average='macro')

if "max_length" in config:
    print("Maximum positional embeddings edited to: ",config["max_length"])

print("Begin Training")
for epoch in range(total_epochs):
    model.train()
    for batch, labels in tqdm.tqdm(train_dl):
        # Modality Dropout
        # Prepare Batch
        if isinstance(list(batch.values())[0],dict):
            for key in batch:
                batch[key] = {k:v.to(device) for k,v in batch[key].items()}
        else:
            batch = {k:v.to(device) for k,v in batch.items()}
        labels = labels.to(device)
        
        # Update Model
        optim.zero_grad()
        logits = model(**batch)
        loss = loss_fn(logits, labels)
        loss.backward()
        optim.step()
        scheduler.step()
        metrics = {"ce_train": loss}
        if not args.n:
            wandb.log(metrics)
        print(loss)
    if (epoch+1)%1 == 0:
        model.eval()
        with torch.no_grad():
            total_labels = []
            total_logits = []
            eval_losses = []
            for batch, labels in tqdm.tqdm(valid_dl):
                # Modality Dropout
                bs = len(labels)
                
                # Prepare Batch
                if isinstance(list(batch.values())[0],dict):
                    for key in batch:
                        batch[key] = {k:v.to(device) for k,v in batch[key].items()}
                else:
                    batch = {k:v.to(device) for k,v in batch.items()}
                labels = labels.to(device)

                # Update Model
                logits = model(**batch)
                eval_loss = loss_fn(logits, labels).detach()
                eval_losses.append(eval_loss.cpu())
                total_logits.append(logits.detach().cpu())
                total_labels.append(labels.detach().cpu())
            total_logits = torch.cat(total_logits).cpu().numpy()
            total_labels = torch.cat(total_labels).cpu().numpy()
            if config['task'] == 'multilabel':
                total_logits = (total_logits > 0).astype('int')
            elif config['task'] == 'multiclass':
                total_logits = np.argmax(total_logits, axis = 1).astype('int')
            total_labels = total_labels.astype('int')
            f1 = metric(total_logits, total_labels)
            metrics = {"f1_valid": f1, "ce_valid": np.average(eval_losses)}
            if not args.n:
                wandb.log(metrics)
            torch.save(model, config["savefile"].format(epoch=epoch))
