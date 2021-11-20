#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:53:11 2021

@author:Yongho Kim
"""
import torch
import torch.nn as nn

class Flatten(torch.nn.Module): 
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1) 
    
class Deflatten(nn.Module): 
    def __init__(self, k):
        super(Deflatten, self).__init__()
        self.k = k
        
    def forward(self, x):
        s = x.size()
        feature_size = int((s[1]//self.k)**.5)       
        return x.view(s[0],self.k,feature_size,feature_size)

class Kmeans(nn.Module): 
    def __init__(self, num_clusters, latent_size):
        super(Kmeans, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_clusters = num_clusters
        self.centroids = nn.Parameter(torch.rand((self.num_clusters, latent_size)).to(device))
    
    def argminl2distance(self, a, b):    
        return torch.argmin(torch.sum((a-b)**2,dim=1),dim=0)

    def forward(self, x):
        y_assign = []
        for m in range(x.size(0)):
            h = x[m].expand(self.num_clusters,-1)
            assign = self.argminl2distance(h, self.centroids)
            y_assign.append(assign.item())
        
        return y_assign, self.centroids[y_assign]

class Encoder(nn.Module):
    def __init__(self, latent_size):
        super(Encoder, self).__init__()
        
        k = 16
        self.encoder = nn.Sequential(
                        nn.Conv2d(1, k, 3, stride=2), 
                        nn.ReLU(), 
                        nn.Conv2d(k, 2*k, 3, stride=2),
                        nn.ReLU(), 
                        nn.Conv2d(2*k, 4*k, 3, stride=1),
                        nn.ReLU(),
                        Flatten(),
                        nn.Linear(1024, latent_size), 
                        nn.ReLU()
        )

    def forward(self, x):       
        return self.encoder(x)
    
class Decoder(nn.Module):
    def __init__(self, latent_size):
        super(Decoder, self).__init__()
        
        k = 16
        self.decoder = nn.Sequential(
                        nn.Linear(latent_size, 1024),
                        nn.ReLU(),
                        Deflatten(4*k),
                        nn.ConvTranspose2d(4*k, 2*k, 3, stride=1), # (입력 채널 수, 출력 채널 수, 필터 크기, stride)
                        nn.ReLU(),
                        nn.ConvTranspose2d(2*k, k, 3, stride=2),
                        nn.ReLU(),
                        nn.ConvTranspose2d(k, 1, 3, stride=2,output_padding=1),
                        nn.Sigmoid()
        )
    
    def forward(self, x):       
        return self.decoder(x)