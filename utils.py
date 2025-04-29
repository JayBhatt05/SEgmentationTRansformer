import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn


########## Helper Dictionary to Visualize Predicted Segmentation Maps ##########
trainId_to_color = {
    0:  (128,  64,128),   # road
    1:  (244,  35,232),   # sidewalk
    2:  ( 70,  70, 70),   # building
    3:  (102, 102,156),   # wall
    4:  (190, 153,153),   # fence
    5:  (153, 153,153),   # pole
    6:  (250, 170, 30),   # traffic light
    7:  (220, 220,  0),   # traffic sign
    8:  (107, 142, 35),   # vegetation
    9:  (152, 251,152),   # terrain
    10: ( 70, 130,180),   # sky
    11: (220,  20, 60),   # person
    12: (255,   0,  0),   # rider
    13: (  0,   0,142),   # car
    14: (  0,   0, 70),   # truck
    15: (  0,  60,100),   # bus
    16: (  0,  80,100),   # train
    17: (  0,   0,230),   # motorcycle
    18: (119,  11, 32),   # bicycle
}


########## Data Transformations ##########
def get_train_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.NEAREST),
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_train_target_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.NEAREST),
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=Image.NEAREST),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))
    ])

def get_val_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_val_target_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.NEAREST),
        transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))
    ])
    
    
########## Loss Function Class including Auxiliary Losses ##########
class SegmentationLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        
    def forward(self, pred, target, aux_outputs=None, aux_weight=0.4):
        loss = self.criterion(pred, target)
        
        # Add auxiliary losses if available
        if aux_outputs is not None:
            for aux_out in aux_outputs:
                loss += aux_weight * self.criterion(aux_out, target)
        
        return loss
    

########## Config Class for Hyperparamter and Model Configuration Setup ##########
# Configuration
class Config:
    # Dataset
    DATASET_ROOT = ''  # Path to the Dataset Directory
    BATCH_SIZE = 4
    NUM_WORKERS = 0
    
    # Model
    MODEL_TYPE = 'SETR-Naive'  # Options: 'SETR-Naive', 'SETR-PUP', 'SETR-MLA' according to the paper
    TRANSFORMER_VARIANT = 'T-Base'  # Options: 'T-Base', 'T-Large'
    PATCH_SIZE = 16
    IMAGE_SIZE = (384, 384)  # Recommended Height, Width for the Cityscapes Dataset (Paper suggested 768 x 768)
    
    # Training
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 0.0001
    MOMENTUM = 0.9
    NUM_EPOCHS = 25  # 80k iterations with batch size 8 mentioned in the paper
    LR_SCHEDULE = 'poly'  # Polynomial LR decay as in the paper
    POWER = 0.9  # Power for polynomial decay
    
    # Cityscapes details
    NUM_CLASSES = 19  # Cityscapes class IDs converted to train IDs (0-18), trainID = 255 assigned to classes that are to be ignored
    
    # Model hyperparameters based on variant
    @staticmethod
    def get_model_config(variant):
        if variant == 'T-Base':
            return {
                'num_layers': 12,
                'hidden_size': 768,
                'num_heads': 12,
                'mlp_ratio': 4
            }
        elif variant == 'T-Large':
            return {
                'num_layers': 24,
                'hidden_size': 1024,
                'num_heads': 16,
                'mlp_ratio': 4
            }
        else:
            raise ValueError(f"Unknown transformer variant: {variant}")