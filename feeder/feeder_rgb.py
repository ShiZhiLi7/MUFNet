# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms

# visualization
import time
import math
# operation
import feeder.tools as tools







class Feeder(torch.utils.data.Dataset):


    def __init__(self,
                 data_path,
                 label_path,
                 temporal_rgb_frames=5,
                 window_size=-1,
                 debug=False,
                 evaluation=False,
                 mmap=True):

        self.debug = debug
        self.evaluation = evaluation
        self.data_path = data_path
        self.label_path = label_path
    
        self.window_size = window_size
     

  
        self.temporal_rgb_frames = temporal_rgb_frames



        self.load_data(mmap)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize(size=224),
            transforms.Resize(size=(225,45*self.temporal_rgb_frames)),
            #transforms.ColorJitter(hue=.05, saturation=.05),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(20, resample=Image.BILINEAR),
            #transforms.RandomErasing(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_evaluation = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize(size=224),
            transforms.Resize(size=(225,45*self.temporal_rgb_frames)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_weight = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=225),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])



    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
            
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]



    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        label = self.label[index]
        sample_name = self.sample_name[index]
        #
        rgb = torch.from_numpy(self.data[index]).float()

        if self.evaluation:
            rgb = self.transform_evaluation(rgb)
        else:
            rgb = self.transform(rgb) # resize to 224x224



        return rgb, label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)