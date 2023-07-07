"""Predict
"""
from modules.utils import load_yaml
from modules.datasets import TestDataset
from models.utils import get_model

from torch.utils.data import DataLoader

from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import numpy as np
import pandas as pd
import random, os, torch
from glob import glob
import sys


# Config
PROJECT_DIR = os.path.dirname(__file__)
predict_config = load_yaml(os.path.join(PROJECT_DIR, 'config', 'predict_config.yaml'))


# Serial
# train_serial = predict_config['TRAIN']['train_serial']
train_serial = sys.argv[1]
kst = timezone(timedelta(hours=9))
predict_timestamp = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")
predict_serial = train_serial + '_' + predict_timestamp

# Predict directory
PREDICT_DIR = os.path.join(PROJECT_DIR, 'results', 'predict', predict_serial)
os.makedirs(PREDICT_DIR, exist_ok=True)

# Recorder Directory
RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', train_serial)

# Data Directory
DATA_DIR = predict_config['DIRECTORY']['dataset']

# Train config
train_config = load_yaml(os.path.join(RECORDER_DIR, 'train_config.yml'))

# SEED
torch.manual_seed(predict_config['PREDICT']['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(predict_config['PREDICT']['seed'])
random.seed(predict_config['PREDICT']['seed'])

model_name = train_config['TRAINER']['model']
backbone = train_config['MODEL'][model_name]['backbone']

# Gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = str(predict_config['PREDICT']['gpu'])
os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[2])

if __name__ == '__main__':

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    X_test = glob(f'{DATA_DIR}/*.png')
    test_dataset = TestDataset(X = X_test, backbone=backbone)
    
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=predict_config['TRAIN']['batch_size'],
                                num_workers=train_config['DATALOADER']['num_workers'], 
                                shuffle=False,
                                pin_memory=train_config['DATALOADER']['pin_memory'],
                                drop_last=False)

    # Load model
    model_name = train_config['TRAINER']['model']
    model_args = train_config['MODEL'][model_name]
    model = get_model(model_name=model_name, model_args=model_args).to(device)

    checkpoint = torch.load(os.path.join(RECORDER_DIR, 'model.pt'))
    model.load_state_dict(checkpoint['model'])

    model.eval()
    
    # Make predictions
    y_preds = []
    y_preds_logits = []
    filenames = []
    for batch_index, (x, filename) in enumerate(tqdm(test_dataloader)):
        x = x.to(device, dtype=torch.float)
        
        y_logits = model(x).squeeze(-1)
        y_preds_logits.append(y_logits.to(torch.float32).detach().cpu())
        y_pred = (y_logits > 0.5).to(torch.int).cpu()
        y_preds.append(y_pred)
        filenames.extend(filename)
    y_preds = torch.cat(y_preds, dim=0).tolist()
    y_preds_logits = torch.cat(y_preds_logits, dim=0).tolist()

    # Save predictions according to the sample submission format
    pred_df = pd.DataFrame({'ImageId':filenames, 'answer': y_preds})
    sample_submission = pd.read_csv(predict_config['DIRECTORY']['sample_submission_path'])
    result = sample_submission.merge(pred_df, on='ImageId', how='left')
    result.drop('answer_x', axis=1, inplace=True)
    result.rename(columns={'answer_y':'answer'}, inplace=True)
    result.to_csv(os.path.join(PREDICT_DIR, 'result.csv'), index=False)
    
    # Save predictions according to the sample submission format
    pred_df = pd.DataFrame({'ImageId':filenames, 'answer': y_preds_logits})
    sample_submission = pd.read_csv(predict_config['DIRECTORY']['sample_submission_path'])
    result = sample_submission.merge(pred_df, on='ImageId', how='left')
    result.drop('answer_x', axis=1, inplace=True)
    result.rename(columns={'answer_y':'answer'}, inplace=True)
    result.to_csv(os.path.join(PREDICT_DIR, 'result_soft.csv'), index=False)

    print('Done')