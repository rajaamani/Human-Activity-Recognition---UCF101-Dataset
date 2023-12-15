import os 
import sys 
import math 

import numpy as np 

from tqdm import tqdm 

import torch 
import torch.nn as nn 

def TrainingStep(model, train_loader, optimizer, criterion, device):

    model.train() 
    Acc = 0.0 
    losses = 0.0 
    count = 0 

    for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        x = x.to(device).float()
        y = y.to(device).long() 

        optimizer.zero_grad() 
        out = model(x) 
        loss  = criterion(out,y) 
        loss.backward() 
        optimizer.step() 
        losses += loss.item() *x.size(0)

        preds = torch.argmax(out, dim=1)

        Acc += (preds == y).sum()  
        count += x.size(0)
    
    losses /= count
    Acc    /= count 
    print(f'Training losses {losses} and Accuracy = {Acc}') 
    return model 


def ValiationStep(model, valid_loader, criterion, device):
    model.eval() 
    Acc = 0.0
    losses = 0.0 
    count = 0 

    for i, (x, y) in tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False):
        x = x.to(device).float()
        y = y.to(device).long()

        with torch.no_grad():
            out = model(x) 
        
        loss  = criterion(out,y)
        losses += loss.item() *x.size(0)
        preds = torch.argmax(out, dim=1)

        Acc += (preds == y).sum()  
        count += x.size(0)
    
    losses /= count 
    Acc    /= count
    print(f'Validation losses {losses} and Accuracy = {Acc}') 



def TrainValidatModel(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs):

    for epoch in range(num_epochs):
        print(epoch,'\n')
        model = TrainingStep(model, train_loader, optimizer=optimizer, criterion=criterion, device=device) 
        ValiationStep(model=model, valid_loader=valid_loader, criterion=criterion, device=device) 
        print('\n')
        