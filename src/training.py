from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import itertools
import random
import spacy

from keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizer
from transformers import BertModel, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from transformers import DistilBertModel, AdamW
from transformers import BertForSequenceClassification

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from pandarallel import pandarallel

import numpy as np
import torch

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def forward_func_baseline(batch, device, model, additional_params):

    # Clear gradients
    model.zero_grad()      
    b_input_ids = batch['input_ids'].to(device)
    b_input_mask = batch['attention_mask'].to(device)
    b_labels = batch['labels'].to(device).float()
    
    loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

    return loss, logits

def forward_func_xlnet(batch, device, model, additional_params):

    # Clear gradients
    model.zero_grad()      
    b_input_ids = batch['input_ids'].to(device)
    b_input_mask = batch['attention_mask'].to(device)
    b_labels = batch['labels'].to(device).float()
    
    loss, logits, _ = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

    return loss, logits

def forward_func_multitask_same_bert(batch, device, model, additional_params):

    # Clear gradients
    model.zero_grad()    
    
    b_input_ids = batch['task_1']['input_ids'].to(device)
    b_input_mask = batch['task_1']['attention_mask'].to(device)
    b_labels = batch['task_1']['labels'].to(device)
    b_pos_tags = batch['task_1']['pos_tag'].to(device)
      
    loss_1, logits = model(b_input_ids, 
                             task_id=1,
                             token_type_ids=None, 
                             pos_tags=b_pos_tags,
                             attention_mask=b_input_mask, 
                             labels=b_labels)
    
    b_input_ids = batch['task_2']['input_ids'].to(device)
    b_input_mask = batch['task_2']['attention_mask'].to(device)
    b_labels = batch['task_2']['labels'].to(device)

    loss_2, _ = model(b_input_ids, 
                             task_id=2,
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

    return 0.85 * loss_1 + 0.15 * loss_2, logits

def forward_func_multitask_bert(batch, device, model, additional_params):

    # Clear gradients
    model.zero_grad()      
    b_input_ids = batch['input_ids'].to(device)
    b_input_mask = batch['attention_mask'].to(device)
        
    if additional_params['task_id'] == 1:
        b_labels = batch['labels'].to(device)
        b_pos_tags = batch['pos_tag'].to(device)
      
        loss, logits = model(b_input_ids, 
                             task_id=1,
                             pos_tags=b_pos_tags,
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)
    elif additional_params['task_id'] == 2:
        b_labels = batch['labels'].to(device)

        loss, logits = model(b_input_ids, 
                             task_id=2,
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)
    return loss, logits

# Function to calculate the accuracy of our predictions vs labels
def flat_metric(preds, labels):
    pred_flat = preds.flatten()
    labels_flat = labels.flatten()
    #return mean_absolute_error(pred_flat, labels_flat)
    return mean_absolute_error(pred_flat, labels_flat)

def forward_func_custom_bert(batch, device, model, additional_params):
    
    b_input_ids = batch['input_ids'].to(device)
    b_input_mask = batch['attention_mask'].to(device)
    b_labels = batch['labels'].to(device)
    b_positions =  batch['target_positions'].to(device)

    # Clear gradients
    model.zero_grad()        

    loss, logits = model(b_input_ids, 
                         b_positions,
                         token_type_ids=None, 
                         attention_mask=b_input_mask, 
                         labels=b_labels)
    
    return loss, logits

def train(device, model, loader, forward_func, optimizer, scheduler, additional_params={}):

    print('Training...')

    total_train_loss = 0
    model.train()

    for step, batch in enumerate(loader):

        loss, logits = forward_func(batch, device, model, additional_params)

        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Compute gradients
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(loader)            

    print("")
    print("  Average training loss: {0:.6f}".format(avg_train_loss))
    
def evaluate(device, model, loader, forward_func, additional_params={}):
    print("")
    print("Running Validation...")
    
    model.eval()
    
    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    
    all_logits = []
    training_stats = []
    # Evaluate data for one epoch
    ix = 0
    for batch in loader:
        ix += 1
        with torch.no_grad():        
            loss, logits = forward_func(batch, device, model, additional_params)
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        
        b_labels = batch['labels'].to(device)
        label_ids = b_labels.to('cpu').numpy()
        
        val_metric = flat_metric(logits, label_ids)
        total_eval_accuracy += val_metric
        all_logits.append(logits.flatten()[0])

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(loader)
    print("  Metric: {0:.6f}".format(avg_val_accuracy))
    
    avg_val_loss = total_eval_loss / len(loader)
    
    print("  Validation Loss: {0:.6f}".format(avg_val_loss))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
        }
    )
    
    return training_stats, all_logits, total_eval_accuracy / len(loader)
