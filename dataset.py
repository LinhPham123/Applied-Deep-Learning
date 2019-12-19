import torch
from torch.utils import data
import numpy as np
import pickle


class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode

    def __getitem__(self, index):
        chroma = self.dataset[index]['features']['chroma']
        spectral_contrast = self.dataset[index]['features']['spectral_contrast']
        tonnetz = self.dataset[index]['features']['tonnetz']

        if self.mode == 'LMC':
            logmel = self.dataset[index]['features']['logmelspec']

            a = np.concatenate((logmel, chroma), axis=0)
            b = np.concatenate((spectral_contrast, tonnetz), axis=0)
            feature = np.concatenate((a, b), axis=0)         
            #feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0) (3D?)
            feature = torch.from_numpy(feature.astype(np.float32))

        elif self.mode == 'MC':
            mfcc = self.dataset[index]['features']['mfcc']

            a = np.concatenate((mfcc, chroma), axis=0)
            b = np.concatenate((spectral_contrast, tonnetz), axis=0)
            feature = np.concatenate((a, b), axis=0)  
            #feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
            feature = torch.from_numpy(feature.astype(np.float32))

        elif self.mode == 'MLMC':
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
       
        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']
        return feature, label, fname

    def __len__(self):
        return len(self.dataset)

load = UrbanSound8KDataset('D:\\Work\\CW\\Deep_Learning\\cw\\UrbanSound8K_test.pkl', 'MC')
a, b, c = load.__getitem__(14)
print(a.shape)
print(a.dim())
print(b)
print(c)
# train_loader = torch.utils.data.DataLoader( 
#       UrbanSound8KDataset('UrbanSound8K_train.pkl', 'LMC'), 
#       batch_size=32, shuffle=True, 
#       num_workers=8, pin_memory=True) 
 
# val_loader = torch.utils.data.DataLoader( 
#      UrbanSound8KDataset('UrbanSound8K_test.pkl', 'MC'), 
#      batch_size=32, shuffle=False, 
#      num_workers=8, pin_memory=True) 

 

# for i, (input, target, filename) in enumerate(train_loader): 
# #           training code
#     print("hello")

# for i, (input, target, filename) in enumerate(val_loader): 
#     print("hi")
# #           validation code 

 