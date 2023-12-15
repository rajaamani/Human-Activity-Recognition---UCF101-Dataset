import os 
import sys 
import math 

import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torch.utils.data as data 

from model.cnnlstm import CNNLSTM
from Data_utils import Generate_TrainData, Generate_ValidationData, generateClassName2ID
from TraionValidation import TrainValidatModel

def Main_functions(model_params=None,  train_params=None, valid_params = None,
                   device = None, batch_size=None, num_workers=None, n_classes=None, num_epochs=30, lr=None):
    
    cls2id, id2cls = generateClassName2ID(fileName='TrainVideos.csv')

    train_data = Generate_TrainData(train_params=train_params, fileName='TrainVideos.csv', cls2id=cls2id, id2cls=id2cls)
    valid_data = Generate_ValidationData(valid_params=valid_params, fileName='ValidVideos.csv', cls2id=cls2id, id2cls=id2cls)
    print(f'length of train datastet {len(train_data)} and validation dataset {len(valid_data)}')
    #print(train_data.cls2id == valid_data.cls2id)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers) 
    valid_loader = data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers) 
    print(f'length of train loader {len(train_loader)} and validation loader {len(valid_loader)}')

    # for x, y in train_data:
    #     print(x.shape, y.shape, x.max(), x.min())
    # for x, y in train_loader:
    #     print(x.shape, y.shape, x.max(), x.min())
        
    # print('Checking the validation dataset')
    # for x, y in valid_loader:
    #     print(x.shape, y.shape, x.max(), x.min())
        
    model = CNNLSTM(num_classes=101)
    model = model.to(device=device)
    criterion = nn.CrossEntropyLoss().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    TrainValidatModel(model=model, train_loader=train_loader, valid_loader=valid_loader, criterion=criterion,
                      optimizer=optimizer, device=device, num_epochs=num_epochs)
    

if __name__ == '__main__':
    os.system('clear') 

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 32 
    num_workers = 4
    model_name = 'resnet'
    n_classes= 101
    lr = 1e-4
    num_epochs = 100
   # mean=[0.4345, 0.4051, 0.3775]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    #std=[0.2768, 0.2713, 0.2737]
    n_input_channels=3
    model_params = {'model_name':model_name,'model_depth':18, 'conv1_t_size':7, 'conv1_t_stride':1, 'resnet_shortcut':'B',
                        'n_input_channels':n_input_channels, 'resnet_widen_factor':1.0, 'no_max_pool':False, 
                        'wide_resnet_k':2,'resnext_cardinality':32}
     

    train_params = {'train_crop':'random', 'mean': mean, 'std':std, 'sample_duration':16, 'train_crop_min_scale':0.25, 
                    'train_crop_min_ratio':0.75, 'no_hflip': False, 'colorjitter':True,'sample_size':112, 'no_mean_norm':False,
                    'no_std_norm':False, 'value_scale':1}
    
    valid_params = {'mean': mean, 'std':std, 'sample_duration':16, 'sample_size':112, 'no_mean_norm':False, 'no_std_norm':False, 
                    'value_scale':1}
     

    Main_functions(model_params=model_params, num_workers=num_workers, n_classes=n_classes, num_epochs=num_epochs,
                   train_params=train_params, valid_params=valid_params, lr=lr, batch_size=batch_size)