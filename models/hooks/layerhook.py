import torch.nn as nn
import torch
class ModelWithHook(nn.Module):
    def __init__(self, model, layer):
        super().__init__()
        self.config = model.config
        self.model = model
        self.features = {}
        layer.register_forward_hook(self.get_features('out'))
 
    def get_features(self, name):
        def hook(model, input, output):
            self.features[name] = output[0].detach()
        return hook

    def forward(self, **x):
        with torch.no_grad():
            output = self.model(**x)
        return self.features['out']
