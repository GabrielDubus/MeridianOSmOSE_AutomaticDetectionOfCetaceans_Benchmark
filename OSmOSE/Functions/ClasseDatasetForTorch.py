# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 20:59:15 2022

@author: gabri
"""



from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os
import scipy.io.wavfile as wav

import sys
with open('path_codes.txt') as f:
    codes_path = f.readlines()[0]
sys.path.append(codes_path)

# CLASS DATASET
class ClassDataset(Dataset):
    def __init__(self, root_dir, annotation_file, param_spectro, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform
        self.nfft = int(param_spectro['nfft'][0])
        self.winsize = int(param_spectro['winsize'][0])
        self.pct_overlap = int(param_spectro['pct_overlap'][0])
        self.Dyn = param_spectro['Dyn']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        file_id = self.annotations.iloc[index, 0]
        fs, audio = wav.read(os.path.join(self.root_dir, file_id))
        y_label = torch.tensor(self.annotations.iloc[index, 1:], dtype=torch.float)
        
        if self.transform is not None:
            spectro = self.transform(audio, fs, self.winsize, self.nfft, self.pct_overlap, self.Dyn, output='torch')

        return (spectro, y_label)
    
    def __getlabels__(self):
        return self.annotations.columns[1:]
    
    def __getfilename__(self, index):
        return self.annotations.iloc[index, 0]
    
