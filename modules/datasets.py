from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms
from glob import glob
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

from albumentations.pytorch import ToTensorV2
# from augmix import RandomAugMix

import albumentations as A

from transformers import EfficientNetImageProcessor, AutoImageProcessor

from modules.utils import TRANSFORMERS_BACKBONE

mean_std = {
    "vit_large_patch14_clip_224.openai_ft_in12k_in1k": {
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711]
    },
    "efficientnet_b0": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "efficientnet_b4": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "resnet50": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "tf_efficientnet_b7.ns_jft_in1k": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "tf_efficientnetv2_l.in21k_ft_in1k": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "tf_efficientnetv2_s.in21k_ft_in1k": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "tf_efficientnetv2_m.in21k_ft_in1k": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "eva02_large_patch14_clip_336.merged2b_ft_inat21": {
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711]
    }
}

AUGMENTATIONS = {
    'custom_hard': lambda: A.Compose([
        A.Resize(height=224, width=224, p=1),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(rotate_limit=0, border_mode=0, p=0.5),
        A.RGBShift(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.CoarseDropout(max_height=24, max_width=24, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2()
    ]),
    'custom_medium': lambda: A.Compose([
        A.Resize(height=224, width=224, p=1),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(rotate_limit=0, border_mode=0, p=0.5),
        A.CoarseDropout(max_height=24, max_width=24, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2()
    ]),
    'custom_soft': lambda: A.Compose([
        A.Resize(height=224, width=224, p=1),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2()
    ]),
    'HT_hard': lambda: A.Compose([
        A.RandomResizedCrop(height=224, width=224, p=1),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=15, p=0.5),
        A.CoarseDropout(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2()
    ]),
    'HT_medium': lambda: A.Compose([
        A.Resize(height=224, width=224, p=1),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=45, p=0.5),
        A.CoarseDropout(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2()
    ]),
    'HT_soft': lambda: A.Compose([
        A.Resize(height=224, width=224, p=1),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2()
    ]),
}



def SplitDataset(img_dir:str, val_size:float=0.1, seed=42, aug_datasets=[]):
    
    img_formats = ['png', 'jpg', 'tif', 'JPG', 'jpeg', 'JPEG']
    
    #original dataset
    fake_images = glob(f'{img_dir}/fake_images/*.png')
    real_images = glob(f'{img_dir}/real_images/*.png')
    
    #aug_datasets
    for aug_dataset in aug_datasets:
        
        for img_format in img_formats:
        
            fake_images+= glob(f'{img_dir}/{aug_dataset}/fake_images/*.{img_format}')
            real_images+= glob(f'{img_dir}/{aug_dataset}/real_images/*.{img_format}')     

    labels = [1] * len(fake_images) + [0] * len(real_images) 

    X_train, X_val, y_train, y_val = train_test_split(fake_images + real_images, labels, test_size=val_size, random_state=seed, shuffle=True)

    return X_train, X_val, y_train, y_val

class CustomDataset(Dataset):
    def __init__(self, X, y, backbone, aug):
        self.X = X
        self.y = y
        self.backbone = backbone
        
        if backbone in TRANSFORMERS_BACKBONE:
            self.processor = AutoImageProcessor.from_pretrained(backbone)
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.Normalize(mean_std[backbone]["mean"], mean_std[backbone]["std"])
            ])
        
        self.aug = aug
        self.custom_aug = self._get_custom_augmentation()
        self.HT_aug = self._get_HT_augmentation()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        target = self.y[index]

        if self.aug in self.custom_aug:
            img = self.custom_aug[self.aug](image=np.array(img))['image']
        elif self.aug in self.HT_aug:
            img = self.HT_aug[self.aug](image=np.array(img))['image']
        else:
            if self.backbone in TRANSFORMERS_BACKBONE:
                img = self.processor(images=img, return_tensors="pt")
                return img['pixel_values'], target, fname
            else:
                img = self.transforms(img)

        return img, target, fname
    
    def _get_custom_augmentation(self):
        return {
            'custom_hard': AUGMENTATIONS['custom_hard'](),
            'custom_medium': AUGMENTATIONS['custom_medium'](),
            'custom_soft': AUGMENTATIONS['custom_soft']()
        }
    
    def _get_HT_augmentation(self):
        return {
            'HT_hard': AUGMENTATIONS['HT_hard'](),
            'HT_medium': AUGMENTATIONS['HT_medium'](),
            'HT_soft': AUGMENTATIONS['HT_soft']()
        }
    
class TestDataset(Dataset):
    def __init__(self, X, backbone):
        '''
        X: list of image path
        '''
        self.X = X
        self.backbone = backbone
        
        if backbone in TRANSFORMERS_BACKBONE:
            self.processor = AutoImageProcessor.from_pretrained(backbone)
        else:
            self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean_std[backbone]["mean"],mean_std[backbone]["std"])
            ])
            
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        impath = self.X[index]
        fname = os.path.basename(impath)
        img = Image.open(impath).convert('RGB')
        if self.backbone in TRANSFORMERS_BACKBONE: #transformer
            img = self.processor(images=img, return_tensors="pt")
            
            return img['pixel_values'],  fname
        
        img = self.transforms(img)

        return img, fname

if __name__ == '__main__':
    pass

        