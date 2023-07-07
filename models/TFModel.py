from torch import nn
from transformers import AutoModel
import timm
import torch

class TFModel(nn.Module):
    def __init__(self, backbone, n_out, is_sigmoid):
        super(TFModel, self).__init__()
        self.model = AutoModel.from_pretrained(backbone)
        
        self.sigmoid = nn.Sigmoid()

        # Head 1
        if backbone == 'umm-maybe/AI-image-detector':
            input_size = 1024
        elif backbone == 'google/vit-large-patch16-224-in21k':
            input_size = 1024
        else:
            input_size = 768

        self.head = nn.Sequential(
            nn.Linear(input_size, n_out)
        )

    def forward(self, x):
        x = torch.squeeze(x)
        x = self.model(x)
        x = x['last_hidden_state'][:, 0, :]      
        x = self.head(x) #head를 통해 1차원으로 축소
        
        
        return x


