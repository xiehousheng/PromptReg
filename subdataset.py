import os
import numpy as np
import torch
from torch.utils.data import Dataset
from medpy.io import load, save
import pandas as pd

def load_nifti(file_path):
    data, header = load(file_path)
    return data

class BaseDataset(Dataset):
    def __init__(self, data_path, split='train', transform=None):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        
        csv_path = os.path.join(data_path, 'csv', f'{split}.csv')
        self.data_pairs = pd.read_csv(csv_path)
    
    def _load_nifti(self, file_path):
        return load_nifti(os.path.join(self.data_path, file_path))
    
    def __len__(self):
        return len(self.data_pairs)

class AbdominalDataset(BaseDataset):
    def _normalize(self, data):
        data = np.clip(data, -1024, 1024)
        return (data - data.min()) / (data.max() - data.min())
    
    def _process_label(self, label):
        unique_values = np.unique(label)
        if not np.all(np.isin(unique_values, np.arange(5))):
            print(f"Warning: Abdominal label contains unexpected values: {unique_values}")
        return label

    def __getitem__(self, idx):
        pair = self.data_pairs.iloc[idx]
        
      
        moving_data = self._load_nifti(pair['moving_image'])
        moving_data = self._normalize(moving_data)
        moving_label = self._load_nifti(pair['moving_label'])
        moving_label = self._process_label(moving_label)
        
    
        fixed_data = self._load_nifti(pair['fixed_image'])
        fixed_data = self._normalize(fixed_data)
        fixed_label = self._load_nifti(pair['fixed_label'])
        fixed_label = self._process_label(fixed_label)
        
     
        moving_tensor = torch.from_numpy(moving_data).float().unsqueeze(0)
        moving_label_tensor = torch.from_numpy(moving_label).long()
        fixed_tensor = torch.from_numpy(fixed_data).float().unsqueeze(0)
        fixed_label_tensor = torch.from_numpy(fixed_label).long()
        
        return {
            'moving': moving_tensor,
            'moving_label': moving_label_tensor,
            'fixed': fixed_tensor,
            'fixed_label': fixed_label_tensor,
            'moving_path': pair['moving_image'],
            'fixed_path': pair['fixed_image']
        }

class BrainDataset(BaseDataset):
 
    def _normalize(self, data):
        p1, p99 = np.percentile(data, (1, 99))
        data = np.clip(data, p1, p99)
        return (data - p1) / (p99 - p1)
    
    def _process_label(self, label):
        unique_values = np.unique(label)
        if not np.all(np.isin(unique_values, np.arange(36))):
            print(f"Warning: Brain label contains unexpected values: {unique_values}")
        return label

    def __getitem__(self, idx):
      
        pair = self.data_pairs.iloc[idx]
        
      
        moving_data = self._load_nifti(pair['moving_image'])
        moving_data = self._normalize(moving_data)
        moving_label = self._load_nifti(pair['moving_label'])
        moving_label = self._process_label(moving_label)
        
      
        fixed_data = self._load_nifti(pair['fixed_image'])
        fixed_data = self._normalize(fixed_data)
        fixed_label = self._load_nifti(pair['fixed_label'])
        fixed_label = self._process_label(fixed_label)
        
     
        moving_tensor = torch.from_numpy(moving_data).float().unsqueeze(0)
        moving_label_tensor = torch.from_numpy(moving_label).long()
        fixed_tensor = torch.from_numpy(fixed_data).float().unsqueeze(0)
        fixed_label_tensor = torch.from_numpy(fixed_label).long()
        
        return {
            'moving': moving_tensor,
            'moving_label': moving_label_tensor,
            'fixed': fixed_tensor,
            'fixed_label': fixed_label_tensor,
            'moving_path': pair['moving_image'],
            'fixed_path': pair['fixed_image']
        }

class CardiacDataset(BaseDataset):
    def _normalize(self, data):
        p1, p99 = np.percentile(data, (1, 99))
        data = np.clip(data, p1, p99)
        return (data - p1) / (p99 - p1)
    
    def _process_label(self, label):
       
        unique_values = np.unique(label)
        if not np.all(np.isin(unique_values, np.arange(4))): 
            print(f"Warning: Cardiact label contains unexpected values: {unique_values}")
        return label

    def __getitem__(self, idx):
      
        pair = self.data_pairs.iloc[idx]
        
      
        moving_data = self._load_nifti(pair['moving_image'])
        moving_data = self._normalize(moving_data)
        moving_label = self._load_nifti(pair['moving_label'])
        moving_label = self._process_label(moving_label)
        
      
        fixed_data = self._load_nifti(pair['fixed_image'])
        fixed_data = self._normalize(fixed_data)
        fixed_label = self._load_nifti(pair['fixed_label'])
        fixed_label = self._process_label(fixed_label)
        
      
        moving_tensor = torch.from_numpy(moving_data).float().unsqueeze(0)
        moving_label_tensor = torch.from_numpy(moving_label).long()
        fixed_tensor = torch.from_numpy(fixed_data).float().unsqueeze(0)
        fixed_label_tensor = torch.from_numpy(fixed_label).long()
        
        return {
            'moving': moving_tensor,
            'moving_label': moving_label_tensor,
            'fixed': fixed_tensor,
            'fixed_label': fixed_label_tensor,
            'moving_path': pair['moving_image'],
            'fixed_path': pair['fixed_image']
        }


class HippocampusDataset(BaseDataset):
   
    def _normalize(self, data):
        p1, p99 = np.percentile(data, (1, 99))
        data = np.clip(data, p1, p99)
        data = (data - data.min()) / (data.max() - data.min())
        return data
    
    def _process_label(self, label):
        unique_values = np.unique(label)
        if not np.all(np.isin(unique_values, [0, 1,2])):
            print(f"Warning: Hippocampus label contains unexpected values: {unique_values}")
        return label

    def __getitem__(self, idx):
       
        pair = self.data_pairs.iloc[idx]
        
     
        moving_data = self._load_nifti(pair['moving_image'])
        moving_data = self._normalize(moving_data)
        moving_label = self._load_nifti(pair['moving_label'])
        moving_label = self._process_label(moving_label)
        
    
        fixed_data = self._load_nifti(pair['fixed_image'])
        fixed_data = self._normalize(fixed_data)
        fixed_label = self._load_nifti(pair['fixed_label'])
        fixed_label = self._process_label(fixed_label)
        
     
        moving_tensor = torch.from_numpy(moving_data).float().unsqueeze(0)
        moving_label_tensor = torch.from_numpy(moving_label).long()
        fixed_tensor = torch.from_numpy(fixed_data).float().unsqueeze(0)
        fixed_label_tensor = torch.from_numpy(fixed_label).long()
        
        return {
            'moving': moving_tensor,
            'moving_label': moving_label_tensor,
            'fixed': fixed_tensor,
            'fixed_label': fixed_label_tensor,
            'moving_path': pair['moving_image'],
            'fixed_path': pair['fixed_image']
        }

class HipDataset(BaseDataset):
 
    def _normalize(self, data):
      
        data=np.clip(data, -1024, 1024)
        return (data - data.min()) / (data.max() - data.min())
    
    def _process_label(self, label):
       
        unique_values = np.unique(label)
        if not np.all(np.isin(unique_values, np.arange(4))):
            print(f"Warning: Hip label contains unexpected values: {unique_values}")
        return label

    def __getitem__(self, idx):
       
        pair = self.data_pairs.iloc[idx]
        
      
        moving_data = self._load_nifti(pair['moving_image'])
        moving_data = self._normalize(moving_data)
        moving_label = self._load_nifti(pair['moving_label'])
        moving_label = self._process_label(moving_label)
        
    
        fixed_data = self._load_nifti(pair['fixed_image'])
        fixed_data = self._normalize(fixed_data)
        fixed_label = self._load_nifti(pair['fixed_label'])
        fixed_label = self._process_label(fixed_label)
        
      
        moving_tensor = torch.from_numpy(moving_data).float().unsqueeze(0)
        moving_label_tensor = torch.from_numpy(moving_label).long()
        fixed_tensor = torch.from_numpy(fixed_data).float().unsqueeze(0)
        fixed_label_tensor = torch.from_numpy(fixed_label).long()
        
        return {
            'moving': moving_tensor,
            'moving_label': moving_label_tensor,
            'fixed': fixed_tensor,
            'fixed_label': fixed_label_tensor,
            'moving_path': pair['moving_image'],
            'fixed_path': pair['fixed_image']
        }

