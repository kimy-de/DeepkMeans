#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:51:48 2021

@author:Yongho Kim
"""

import data
import model as m
import utils
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep k-Means')
    parser.add_argument('--dataset', default='mnist', type=str, help='datasets')
    parser.add_argument('--mode', default='train', type=str, help='train or eval')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')   
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs')  
    parser.add_argument('--num_clusters', default=10, type=int, help='num of clusters') 
    parser.add_argument('--latent_size', default=10, type=int, help='size of latent vector') 
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--lam', default=1e-2, type=float, help='final rate of clustering loss')
    parser.add_argument('--anls', default=10, type=int, help='annealing start point of lambda')
    parser.add_argument('--anle', default=110, type=int, help='annealing end point of lambda')
    parser.add_argument('--pret', default=None, type=str, help='pretrained model path')
                 

    args = parser.parse_args()
    print(args)
    
    # CPU/GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'{device} is available.')

    # Load a dataset
    trainloader, testloader = data.datasets(args.dataset, args.batch_size)
    
    # Set a model
    encoder = m.Encoder(args.latent_size).to(device)
    decoder = m.Decoder(args.latent_size).to(device)
    kmeans = m.Kmeans(args.num_clusters, args.latent_size).to(device)
    if args.pret != None:
        encoder.load_state_dict(torch.load(args.pret+'en.pth'))
        decoder.load_state_dict(torch.load(args.pret+'de.pth'))
        kmeans.load_state_dict(torch.load(args.pret+'clt.pth'))
    
    if args.mode == "train":
        # Loss and optimizer
        criterion1 = torch.nn.MSELoss()
        criterion2 = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(encoder.parameters()) + 
                                     list(decoder.parameters()) +
                                     list(kmeans.parameters()), lr=args.lr)
        
        # Training
        T1 = args.anls
        T2 = args.anle
        ls = 0.0183
        for ep in range(args.epochs):
            if (ep > T1) and (ep < T2):
                alpha = args.lam*(ep - T1)/(T2 - T1) # 1/100, 2/100, .., 99/100
            elif ep >= T2:    
                alpha = args.lam
            else:
                alpha = args.lam/(T2 - T1)
                
            running_loss = 0.0
            for images, _ in trainloader:
                inputs = images.to(device)
                optimizer.zero_grad()
                latent_var = encoder(inputs)
                _, centroids = kmeans(latent_var.detach())
                outputs = decoder(latent_var)
                
                l_rec = criterion1(inputs, outputs) 
                l_clt = criterion2(latent_var, centroids) 
                loss = l_rec + alpha*l_clt
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
          
            avg_loss = running_loss / len(trainloader)        
            
            if ep % 10 == 0:               
                testacc = utils.evaluation(testloader, encoder, kmeans, device)
                print('[%d] Train loss: %.4f, Test Accuracy: %.3f' %(ep, avg_loss, testacc))  
                
            if avg_loss < ls:
                ls = avg_loss
                torch.save(encoder.state_dict(),'./_en.pth')
                torch.save(decoder.state_dict(),'./_de.pth')
                torch.save(kmeans.state_dict(),'./_clt.pth')
        
        
        # Final Test
        testacc = utils.evaluation(testloader, encoder, kmeans, device)
        print('Test Accuracy: %.3f' %(testacc)) 
             
        