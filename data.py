#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:52:55 2021

@author:Yongho Kim
"""
import torch
import torchvision
from torchvision import transforms


def datasets(data_name, batch_size):
    
    if data_name == "mnist":
        trainset = torchvision.datasets.MNIST('./data/', download=True, train=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST('./data/', download=True, train=False, transform=transforms.ToTensor())
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
        
    elif data_name == "ns":
        pass
     
    return trainloader, testloader
