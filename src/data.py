from xml.dom import minidom
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd
import numpy as np
import torch

class ComposeMultiDataset(torch.utils.data.Dataset):
    def __init__(self, encodings_dis, labels_dis, encodings_lex, labels_lex, positions_lex, pos_tags_lex):
        self.encodings_dis = encodings_dis
        self.labels_dis = labels_dis
        
        self.encodings_lex = encodings_lex
        self.labels_lex = labels_lex
        self.pos_tags_lex = pos_tags_lex
        self.positions_lex = positions_lex

    def __getitem__(self, idx):
        item = {}
        
        task_1 = {key: torch.tensor(val[idx]) for key, val in self.encodings_lex.items()}
        task_1['labels'] = torch.tensor(self.labels_lex[idx])
        task_1['pos_tag'] = torch.tensor(self.pos_tags_lex[idx])
        
        task_2 = {key: torch.tensor(val[idx]) for key, val in self.encodings_dis.items()}
        task_2['labels'] = torch.tensor(self.labels_dis[idx])
        
        item['task_2'] = task_2
        item['task_1'] = task_1
        return item

    def __len__(self):
        return len(self.labels_dis)
    
class ComposeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, positions, pos_tags):
        self.encodings = encodings
        self.labels = labels
        self.positions = positions
        self.pos_tags = pos_tags

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['pos_tag'] = torch.tensor(self.pos_tags[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
class LexDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, positions):
        self.encodings = encodings
        self.labels = labels
        self.positions = positions

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['target_positions'] = torch.tensor(self.positions[idx])
        return item

    def __len__(self):
        return len(self.labels)