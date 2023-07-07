from torch import nn
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification

class EffNetGoogle(nn.Module):
    def __init__(self, backbone, n_out, is_sigmoid):
        super(EffNetGoogle, self).__init__()
        self.model = EfficientNetForImageClassification.from_pretrained(backbone)
        self.model.classifier = nn.LazyLinear(n_out)
        self.is_sigmoid = is_sigmoid

    def forward(self, x):
        x = self.model(x)
        if self.is_sigmoid:
            x = nn.Sigmoid()(x.logits)
        return x

