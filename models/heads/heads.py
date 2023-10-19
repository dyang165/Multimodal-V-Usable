import torch.nn as nn
import torch

class MultiLabelClassification(nn.Module):
    def __init__(self, num_classes, hidden_dim=768):
        super().__init__()
        self.proj1 = nn.Linear(hidden_dim, hidden_dim) 
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.nonlinearity = nn.GELU()
        self.proj2 = nn.Linear(hidden_dim, num_classes)
 
    def forward(self,x):
        x = self.proj1(x)
        x = self.layernorm(x)
        x = self.nonlinearity(x)
        x = self.proj2(x)
        return x


class MultiLabelClassificationWithUpProj(nn.Module):
    def __init__(self, num_classes, hidden_dim=768):
        super().__init__()
        self.proj1 = nn.Linear(hidden_dim, hidden_dim * 2) 
        self.layernorm = nn.LayerNorm(hidden_dim * 2)
        self.nonlinearity = nn.GELU()
        self.proj2 = nn.Linear(hidden_dim * 2, num_classes)
 
    def forward(self,x):
        x = self.proj1(x)
        x = self.layernorm(x)
        x = self.nonlinearity(x)
        x = self.proj2(x)
        return x


class MultiLabelClassificationWithUpProjAndPooler(nn.Module):
    def __init__(self, num_classes, hidden_dim=768):
        super().__init__()
        self.pooler_dense = nn.Linear(hidden_dim, hidden_dim)
        self.pooler_activation = nn.Tanh()
        self.head_proj1 = nn.Linear(hidden_dim, 2*hidden_dim) 
        self.head_layernorm = nn.LayerNorm(2*hidden_dim)
        self.head_nonlinearity = nn.GELU()
        self.head_proj2 = nn.Linear(2*hidden_dim, num_classes)
 
    def forward(self, hidden_states):
        x = hidden_states[:, 0]
        x = self.pooler_dense(x)
        x = self.pooler_activation(x)
        x = self.head_proj1(x)
        x = self.head_layernorm(x)
        x = self.head_nonlinearity(x)
        x = self.head_proj2(x)
        return x
