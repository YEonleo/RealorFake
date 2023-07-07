from torch import nn
from transformers import AutoModel, CLIPVisionModel
import timm
import torch

class CLIPModel(nn.Module):
    def __init__(self, backbone, n_out, is_sigmoid):
        super(CLIPModel, self).__init__()
        self.model = CLIPVisionModel.from_pretrained(backbone)
        
        self.sigmoid = nn.Sigmoid()

        # Head 1
        if backbone == 'laion/CLIP-ViT-g-14-laion2B-s12B-b42K':
            input_size = 1408

        self.head = nn.Sequential(
            nn.Linear(input_size, n_out)
        )

    def forward(self, x):
        x = torch.squeeze(x)
        x = self.model(pixel_values=x)
        x = x['last_hidden_state'][:, 0, :]      
        x = self.head(x) #head를 통해 1차원으로 축소
        
        
        return x


