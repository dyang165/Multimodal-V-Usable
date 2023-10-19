from torch import nn
import torch
from transformers import AutoModel, AutoConfig
import transformers
import inspect

class MergeProcessorVLLF(nn.Module):
    def __init__(self, text_processor, image_processor):
        super().__init__()
        self.text_processor = text_processor
        self.image_processor = image_processor  
 
    def __call__(self,**x):
        text_inputs = self.text_processor(text = x['text'], max_length=512, padding = 'max_length', truncation = True, return_tensors='pt')
        image_inputs = self.image_processor(images = x['images'], return_tensors='pt')  
        return {'text_inputs':text_inputs, 'image_inputs':image_inputs}

class MergeProcessorDoubleVilt(nn.Module):
    def __init__(self, text_processor, image_processor, config):
        super().__init__()
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.config = config 

    def __call__(self,**x):
        text_inputs = self.text_processor(text = x['text'],images=x['images'], max_length=self.config["max_length"], padding = 'max_length', truncation = True, return_tensors='pt')
        image_inputs = self.image_processor(text = x['text'],images = x['images'], max_length = self.config["max_length"], padding = "max_length", truncation=True, return_tensors='pt')  
        return {'text_inputs':text_inputs, 'image_inputs':image_inputs}



class Output():
    pooler_output: torch.FloatTensor = None
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output
 
class VLLF(nn.Module):
    def __init__(self, text_pt, img_pt):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_pt)
        self.img_encoder = AutoModel.from_pretrained(img_pt)
        self.config = self.text_encoder.config
        self.config.hidden_size = self.config.hidden_size + self.img_encoder.config.hidden_size

    def forward(self, text_inputs, image_inputs, *kargs):
        text_outputs = self.text_encoder(**text_inputs).pooler_output
        img_outputs = self.img_encoder(**image_inputs).pooler_output
        concat = torch.cat([text_outputs, img_outputs],dim=1)
        return Output(pooler_output = concat)


class VLLFDoubleVilt(nn.Module):
    def __init__(self,  config):
        super().__init__()
        text_state_dict = torch.load(config['text_checkpoint']).module.model.state_dict()
        img_state_dict = torch.load(config['image_checkpoint']).module.model.state_dict()
        self.text_encoder = transformers.ViltModelNoImg.from_pretrained(config['text_pt_checkpoint'])
        self.img_encoder = transformers.ViltModelNoText.from_pretrained(config['image_pt_checkpoint'])
        if "max_length" in config:
            self.extend_pos_embeds(self.text_encoder, config["max_length"])
            self.extend_pos_embeds(self.img_encoder, config["max_length"])
        self.text_encoder.load_state_dict(text_state_dict)
        self.img_encoder.load_state_dict(img_state_dict)
        if 'freeze' in config and config['freeze'] == True:
            self.freeze(self.text_encoder)
            self.freeze(self.img_encoder)

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False        

    def extend_pos_embeds(self, model, max_length):
        embedding_size = model.config.hidden_size
        pretrained_pos_emb=model.embeddings.text_embeddings.position_embeddings
        extended_pos_emb = torch.cat([pretrained_pos_emb.weight for _ in range(0, max_length, 40)], 0)
        model.embeddings.text_embeddings.position_embeddings=nn.Embedding(max_length, embedding_size).from_pretrained(    extended_pos_emb, freeze=False)
        model.embeddings.text_embeddings.register_buffer("position_ids", torch.arange(max_length).expand((1, -1)))


    def forward(self, text_inputs, image_inputs, *kargs):
        text_outputs = torch.mean(self.text_encoder(**text_inputs).last_hidden_state, dim=1)
        img_outputs = torch.mean(self.img_encoder(**image_inputs).last_hidden_state, dim=1)
        concat = torch.cat([text_outputs, img_outputs],dim=1)
        return Output(pooler_output = concat)

