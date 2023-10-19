import torch 
import torch.nn as nn
import transformers

class ResNet50ModelWrapper(nn.Module):
    def __init__(self,path):
        super().__init__()
        self.model = transformers.ResNetModel.from_pretrained(path)
        self.config = self.model.config.to_dict()
        self.config['hidden_size'] = 2048

    def forward(self, **x):
        output = self.model(**x)
        output.pooler_output = output.pooler_output.squeeze()
        return output
