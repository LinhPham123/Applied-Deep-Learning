import torch
from torch.utils import data
import numpy as np
import pickle


class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        self.mode = mode

    def __getitem__(self, index):
        mfcc = self.dataset[index]['features']['mfcc']
        logmel = self.dataset[index]['features']['logmelspec']
        chroma = self.dataset[index]['features']['chroma']
        spectral_contrast = self.dataset[index]['features']['spectral_contrast']
        tonnetz = self.dataset[index]['features']['tonnetz']

        if self.mode == 'LMC':     
            feature = np.concatenate((logmel, chroma, spectral_contrast, tonnetz), axis=0)      
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0) #size [1, 85, 41]
    
        elif self.mode == 'MC':
            feature = np.concatenate((mfcc, chroma, spectral_contrast, tonnetz), axis=0)
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0) #size [1, 85, 41]

        elif self.mode == 'MLMC':
            feature = np.concatenate((mfcc, logmel, chroma, spectral_contrast, tonnetz), axis=0)   
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0) #size [1, 145, 41]
       
        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']
        return feature, label, fname

    def __len__(self):
        return len(self.dataset)