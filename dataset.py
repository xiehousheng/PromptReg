import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import threading
from subdataset import AbdominalDataset, BrainDataset, CardiacDataset, HippocampusDataset, HipDataset

class MultiTaskRegistrationDataset(Dataset):
    def __init__(self, data_root, target_size=(160, 160, 160), split='train', exclude_tasks=None):
       
        self.data_root = data_root
        self.target_size = np.array(target_size)
        self.split = split
        
      
        self.task_map = {
            'Abdominal': 0,   
            'Brain': 1,    
            'Hippocampus': 3,   
            'Cardiac': 4,  
            'Hip': 5     
        }
        
        self._full_num_classes = {
            'Abdominal': 5,    # Abdominal
            'Brain': 36,  # Brain
            'Hippocampus': 3,   # Hippocampus
            'Cardiac': 4,   # Cardiac
            'Hip': 4      # Hip
        }
        
      
        if exclude_tasks:
            for task in exclude_tasks:
                if task in self.task_map:
                    del self.task_map[task]
            
         
            new_id = 0
            new_task_map = {}
            for task, _ in self.task_map.items():
                new_task_map[task] = new_id
                new_id += 1
            self.task_map = new_task_map
        
    
        self.num_classes = {}
        for task, task_id in self.task_map.items():
            self.num_classes[task_id] = self._full_num_classes[task]
        
   
        self.datasets = {}
        if 'Abdominal' in self.task_map:
            self.datasets[self.task_map['Abdominal']] = AbdominalDataset(os.path.join(data_root, 'Abdominal'), split=split)
        if 'Brain' in self.task_map:
            self.datasets[self.task_map['Brain']] = BrainDataset(os.path.join(data_root, 'Brain'), split=split)
        if 'Cardiac' in self.task_map:
            self.datasets[self.task_map['Cardiac']] = CardiacDataset(os.path.join(data_root, 'Cardiac'), split=split)
        if 'Hippocampus' in self.task_map:
            self.datasets[self.task_map['Hippocampus']] = HippocampusDataset(os.path.join(data_root, 'Hippocampus'), split=split)
        if 'Hip' in self.task_map:
            self.datasets[self.task_map['Hip']] = HipDataset(os.path.join(data_root, 'Hip'), split=split)
        
    
        self.dataset_sizes = {task_id: len(dataset) for task_id, dataset in self.datasets.items()}
        print("Dataset sizes:", self.dataset_sizes)
        print("Task map:", self.task_map)
        print("Number of classes per task:", self.num_classes)
        
    
        self.task_num = len(self.task_map)
        self.task_pool = [100] * self.task_num 
        self.epoch_choice_id = list(range(self.task_num))
        self.lock = threading.Lock()


    def get_task_id(self, name):
        for task_name, task_id in self.task_map.items():
            if name.startswith(task_name):
                return task_id
        raise ValueError(f"Unknown task for data: {name}")
    
    def get_task_name(self, task_id):
      
        for task_name, tid in self.task_map.items():
            if tid == task_id:
                return task_name
        raise ValueError(f"Unknown task ID: {task_id}")

    def reset_task_pool(self):

        self.task_pool = [100] * self.task_num
        self.epoch_choice_id = list(range(self.task_num))

    def resample_to_target_size(self, image, label=None):

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
                
            

        if label is not None and isinstance(label, np.ndarray):
            label = torch.from_numpy(label)
            
   
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), 
            size=tuple(self.target_size),
            mode='trilinear',
            align_corners=True
        ).squeeze(0)  
        
        if label is not None:
         
            original_unique = torch.unique(label)
         
            label = label.float() 
            label = torch.nn.functional.interpolate(
                label.unsqueeze(0).unsqueeze(0), 
                size=tuple(self.target_size),
                mode='nearest'
            ).squeeze(0).squeeze(0) 
            
         
            label = label.long()
           
            resampled_unique = torch.unique(label)
          
            return image, label
        
        return image

    def __getitem__(self, idx):
      
        with self.lock:
        
            choice_id = np.random.choice(self.epoch_choice_id)
            self.task_pool[choice_id] -= 1

         
            if self.task_pool[choice_id] == 0:
                if len(self.epoch_choice_id) == 1:
                 
                    self.reset_task_pool()
                else:
                 
                    self.epoch_choice_id.remove(choice_id)
 
        dataset = self.datasets[choice_id]
        idx = np.random.randint(len(dataset))
        data = dataset[idx]
        
        
        moving, moving_label = self.resample_to_target_size(data['moving'], data['moving_label'])
        fixed, fixed_label = self.resample_to_target_size(data['fixed'], data['fixed_label'])
        

        
     
        original_size = data['moving'].shape[1:]  # [H, W, D]
        
     
        task_name = self.get_task_name(choice_id)
    
        return {
            'moving': moving,           # [1, H, W, D]
            'moving_label': moving_label,  # [H, W, D]
            'fixed': fixed,             # [1, H, W, D]
            'fixed_label': fixed_label,    # [H, W, D]
            'task_id': choice_id,
            'task_name': task_name,   
            'moving_path': data['moving_path'],
            'fixed_path': data['fixed_path'],
            'original_size': original_size  
        }

    def __len__(self):
        return 100 * self.task_num

    def get_dataloader(self, batch_size, num_workers=4):
        def _worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            worker_init_fn=_worker_init_fn
        )

    def get_task_dataset(self, task_id):
        if task_id not in self.datasets:
            raise ValueError(f"Invalid task_id: {task_id}")
            
        class TaskDataset(Dataset):
            def __init__(self, parent_dataset, task_id):
                self.parent = parent_dataset
                self.task_id = task_id
                self.dataset = parent_dataset.datasets[task_id]
                
            def __getitem__(self, idx):
                data = self.dataset[idx]
             
                moving, moving_label = self.parent.resample_to_target_size(data['moving'], data['moving_label'])
                fixed, fixed_label = self.parent.resample_to_target_size(data['fixed'], data['fixed_label'])
                
            
                task_name = self.parent.get_task_name(self.task_id)
                
                return {
                    'moving': moving,
                    'moving_label': moving_label,
                    'fixed': fixed,
                    'fixed_label': fixed_label,
                    'task_id': self.task_id,
                    'task_name': task_name,    
                    'moving_path': data['moving_path'],
                    'fixed_path': data['fixed_path'],
                    'original_size': data['moving'].shape[1:]
                }
                
            def __len__(self):
                return len(self.dataset)
                
        return TaskDataset(self, task_id)
