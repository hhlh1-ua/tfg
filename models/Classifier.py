import torch.nn as nn
import torch.nn.functional as F
import torch

class Classifier (nn.Module):
    def __init__(self, hidden_dims=[256, 128, 64], input_dim=768,output_dim=32, dropout=0.2):
        super().__init__()
        # hidden_dims = config.get('classifier', {}).get('hidden_dims', [256, 128, 64])
        layers = []
        current_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            layers.append(nn.LayerNorm(dim))
            current_dim = dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        # layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)
        

    def forward(self, x):
        return self.model(x)




