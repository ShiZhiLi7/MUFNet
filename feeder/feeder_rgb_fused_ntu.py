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

# rgb --B
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True





class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 rgb_path,

                 debug=False,
                 evaluation=False,
                 mmap=False):
        self.debug = debug
        self.evaluation = evaluation
        self.data_path = data_path
        self.label_path = label_path
        self.rgb_data_path = rgb_path

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize(size=224),
            transforms.Resize(size=(225, 225)),
            # transforms.ColorJitter(hue=.05, saturation=.05),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(20, resample=Image.BILINEAR),
            # transforms.RandomErasing(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_evaluation = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize(size=224),
            transforms.Resize(size=(225, 225)),
            # transforms.ToTensor(),

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])



        self.load_data(mmap)



    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)




        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
            self.rgb_data = np.load(self.rgb_data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            self.rgb_data = np.load(self.rgb_data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.rgb_data = self.rgb_data[0:100]
            self.sample_name = self.sample_name[0:100]
    
        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.rgb_data)



    def __getitem__(self, index):
       
        label = self.label[index]
        sample_name = self.sample_name[index]
        rgb_numpy = self.rgb_data[index]
        # get data
        data_numpy = self.data[index]


        # if self.evaluation:
        #     rgb_numpy = self.transform_evaluation(rgb_numpy)
        # else:
        #     rgb_numpy = self.transform(rgb_numpy)  # resize to 224x224


        return sample_name,label,data_numpy, rgb_numpy
    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
