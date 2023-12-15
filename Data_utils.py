import os 
import sys 
import math 

import numpy as np 
import pandas as pd 

import torch 
import torch.nn as nn 
import torch.nn.functional as F  
import torch.utils.data as data 

import torchvision.transforms.transforms as transforms 

from Spatial_transforms import MultiScaleCornerCrop

from temporal_transforms import LoopPadding, TemporalBeginCrop, TemporalCenterCrop, TemporalRandomCrop

from dataset.VIdeoLoader import VideoDataset


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return [0, 0, 0], [1, 1, 1]
        else:
            return [0, 0, 0], std
    else:
        if no_std_norm:
            return mean, [1, 1, 1]
        else:
            return mean, std

def generateClassName2ID(fileName='TrainVideos.csv'):
    data_df = pd.read_csv(fileName)

    data_df.columns = ['Index', 'videoName']
    clsName2ID = {} 
    ID2clsName = {} 
    clsNames = [] 
    i = 0 

    for video in data_df.iterrows():
        videoName = video[1][1]
        clsName = videoName.split('/')[2]

        if clsName not in clsNames:
            clsNames.append(clsName) 
            clsName2ID[clsName] = i 
            ID2clsName[i] = clsName 
            i = i+1

    return clsName2ID, ID2clsName

def Generate_TrainData(train_params = None, fileName='TrainVideos.csv', cls2id=None, id2cls=None):
    assert train_params['train_crop'] in ['random', 'corner', 'center']
    spatial_transform = []
    if train_params['train_crop'] == 'random':
        spatial_transform.append(transforms.RandomResizedCrop(train_params['sample_size'], (train_params['train_crop_min_scale'], 1.0),
                (train_params['train_crop_min_ratio'], 1.0 / train_params['train_crop_min_ratio'])))
    elif train_params['train_crop'] == 'corner':
        scales = [1.0]
        scale_step = 1 / (2**(1 / 4))
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(MultiScaleCornerCrop(train_params['sample_size'], scales))
    elif train_params['train_crop'] == 'center':
        spatial_transform.append(transforms.Resize(train_params['sample_size']))
        spatial_transform.append(transforms.CenterCrop(train_params['sample_size']))
    
    mean, std = get_normalize_method(train_params['mean'], train_params['std'], train_params['no_mean_norm'],
                                      train_params['no_std_norm'], )
    print(f'train mean {mean} and Variance {std}')
    spatial_transform.append(transforms.RandomHorizontalFlip())
    spatial_transform.append(transforms.ColorJitter())
    spatial_transform.append(transforms.ToTensor())
    spatial_transform.append(transforms.Normalize(mean=mean, std=std)) 
    spatial_transform = transforms.Compose(spatial_transform)

    # temporal_transform = RandomTemporalCrop(size=train_params['sample_duration'])
    
    # temporal_transform = TemporalCompose(temporal_transform)

    train_data = VideoDataset(spatial_transform=spatial_transform, temporal_transform=None,
                              size=train_params['sample_duration'], cls2id=cls2id, id2cls=id2cls)
    return train_data  

def Generate_ValidationData(valid_params = None, fileName='ValidVideos.csv', cls2id=None, id2cls=None):
    mean, std = get_normalize_method(valid_params['mean'], valid_params['std'], valid_params['no_mean_norm'],
                                      valid_params['no_std_norm'])
    
    spatial_transform = [ transforms.Resize(valid_params['sample_size']),
                          #transforms.CenterCrop(valid_params['sample_size']),
                          transforms.ToTensor(), 
                          transforms.Normalize(mean=mean, std=std)]
    spatial_transform = transforms.Compose(spatial_transform) 

    validation_data = VideoDataset(fileName='ValidVideos.csv', train=False, temporal_transform=None, 
                                   spatial_transform=spatial_transform, size=valid_params['sample_duration'],
                                   cls2id=cls2id, id2cls=id2cls)
    return validation_data