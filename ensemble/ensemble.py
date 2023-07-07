"""ensemble
"""

import yaml
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import random, os


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

# Config
ENSEMBLE_DIR = os.path.dirname(__file__)
ensemble_config = load_yaml(os.path.join(ENSEMBLE_DIR, 'ensemble_config.yaml'))


# Ensemble directory
ENSEMBLE_DIR = os.path.join(ENSEMBLE_DIR, 'ensemble_results')

if not os.path.isdir(ENSEMBLE_DIR):
    os.makedirs(ENSEMBLE_DIR)


ENSEMBLE_DIR = os.path.join(ENSEMBLE_DIR)
os.makedirs(ENSEMBLE_DIR, exist_ok=True)

INFERENCE_DIR = os.path.dirname(__file__)
INFERENCE_DIR = os.path.join(INFERENCE_DIR, "each_inferences")

if ensemble_config['IS_PROB']:
    INFERENCE_DIR = os.path.join(INFERENCE_DIR, "soft")
else:
    INFERENCE_DIR = os.path.join(INFERENCE_DIR, "label") 

csv_files = os.listdir(INFERENCE_DIR)


def soft_voting(voters):
    sample_submission = pd.read_csv(ensemble_config['DIRECTORY']['sample_submission_path'])
    
    if ensemble_config['IS_PROB']:
        result_filename = "soft_"
    else:
        result_filename = ""
    
    result_filename+= ensemble_config['VOTING_METHOD']+"_"
    
    #zeros
    ensemble_pred = np.zeros(len(sample_submission), dtype=float)
    
    for voter in voters:
        result_filename+=voter+"+"
        
        if not os.path.isfile(os.path.join(INFERENCE_DIR, voters[voter]['csv_file'])):
            print("File not found : ", os.path.join(INFERENCE_DIR, voters[voter]['csv_file']))
        
        #load csv file
        voter_df = pd.read_csv(os.path.join(INFERENCE_DIR, voters[voter]['csv_file']))
        voter_prob = voter_df['answer']
        filenames = voter_df['ImageId']
        
        #add probs to zeros (soft voting)~
        ensemble_pred+=np.array(voter_prob)
        
    #divide to number of voters
    ensemble_pred /= len(voters)
    ensemble_pred= ensemble_pred.tolist()
    
    for i in range(len(ensemble_pred)):
        ensemble_pred[i] = int(ensemble_pred[i] > 0.5)
            
    ensemble_df = pd.DataFrame({'ImageId':filenames, 'answer':ensemble_pred})
    result = sample_submission.merge(ensemble_df, on='ImageId', how='left')
    result.drop('answer_x', axis=1, inplace=True)
    result.rename(columns={'answer_y':'answer'}, inplace=True)
    result.to_csv(os.path.join(ENSEMBLE_DIR, result_filename+".csv"), index=False)
    
    
    
def weighted_voting(voters):
    sample_submission = pd.read_csv(ensemble_config['DIRECTORY']['sample_submission_path'])
    result_filename = ensemble_config['VOTING_METHOD']+"_"
    
    if ensemble_config['IS_PROB']:
        result_filename = "soft_"
    else:
        result_filename = ""
    
    result_filename+= ensemble_config['VOTING_METHOD']+"_"
    
    #zeros
    ensemble_pred = np.zeros(len(sample_submission), dtype=float)
    
    #for weighted vote
    total_macrof1 = 0
    
    for voter in voters:
        total_macrof1+=voters[voter]['macrof1']
        
        result_filename+=voter+"+"
        
        if not os.path.isfile(os.path.join(INFERENCE_DIR, voters[voter]['csv_file'])):
            print("File not found : ", os.path.join(INFERENCE_DIR, voters[voter]['csv_file']))
        
        #load csv file
        voter_df = pd.read_csv(os.path.join(INFERENCE_DIR, voters[voter]['csv_file']))
        voter_prob = voter_df['answer']
        filenames = voter_df['ImageId']
        
        #add probs to zeros (soft voting)~
        ensemble_pred+= np.array(voter_prob) * voters[voter]['macrof1']

        
    #divide to number of total macrof1
    ensemble_pred /= total_macrof1
    ensemble_pred= ensemble_pred.tolist()
    
    for i in range(len(ensemble_pred)):
        ensemble_pred[i] = int(ensemble_pred[i] > 0.5)
            
    ensemble_df = pd.DataFrame({'ImageId':filenames, 'answer':ensemble_pred})
    result = sample_submission.merge(ensemble_df, on='ImageId', how='left')
    result.drop('answer_x', axis=1, inplace=True)
    result.rename(columns={'answer_y':'answer'}, inplace=True)
    result.to_csv(os.path.join(ENSEMBLE_DIR, result_filename+".csv"), index=False)
    
    print("Filename: ", result_filename+".csv")
    


voting = {
    'soft':soft_voting,
    # 'hard':hard_voting,
    'weighted':weighted_voting
}

if __name__ == '__main__':
    
    voters = ensemble_config['VOTERS']
    voting[ensemble_config['VOTING_METHOD']](voters)
    

    