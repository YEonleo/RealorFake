from models.effnet import EffNet
from models.effnet_google import EffNetGoogle
from models.TFModel import TFModel
from models.CLIPModel import CLIPModel

def get_model(model_name:str, model_args:dict):
    if model_name == 'effnet':
        return EffNet(**model_args)
    
    elif model_name == 'effnet_google':
        return EffNetGoogle(**model_args)
    
    elif model_name == 'transformers':
        return TFModel(**model_args)
    
    elif model_name == 'AIorNot':
        return TFModel(**model_args)
    
    elif model_name == 'AIorNot2':
        return TFModel(**model_args)
    
    elif model_name == 'GVit':
        return TFModel(**model_args)
    
    elif model_name == 'GVit_L':
        return TFModel(**model_args)
    elif model_name == 'CLIP_Vit':
        return CLIPModel(**model_args)
    
    else: 
        raise ValueError(f'Model name {model_name} is not valid.')

if __name__ == '__main__':
    pass