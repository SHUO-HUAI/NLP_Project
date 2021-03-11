import argparse
import os
from os import listdir
from os.path import isfile, join
from io import open
import torch
from torch.nn.utils.rnn import pack_padded_sequence as pack
import sys
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from collections import Counter
from torch.autograd import Variable
import time
import classes
import preprocessing
import model
import functions
from functions import to_cuda
from preprocessing import read_files, prepare_data, prepare_summary, zero_pad
from model import Model
import config

parser = argparse.ArgumentParser(description='PyTorch Get To The Point Training')
parser.add_argument('--path', type=str, default=config.path,
                    help='path to the training data')

args = parser.parse_args()

path = args.path

articles, summaries, dic = read_files(path)

word_count = len(dic)
print('Number of unique words:', word_count)

art_idx = prepare_data(articles, dic)
sum_idx = prepare_summary(summaries, dic)

padded_articles = zero_pad(art_idx)
padded_summaries = zero_pad(sum_idx)

print('Length of padded articles:', len(padded_articles[0]))
print('Length of padded summaries:', len(padded_summaries[0]))

tensor_art = torch.LongTensor(padded_articles)
tensor_sum = torch.LongTensor(padded_summaries)

articles_len = len(tensor_art[0])

model = Model(dic, articles_len)
model = to_cuda(model)


# TEMPORARY CODE
# TEMPORARY CODE
# TEMPORARY CODE

opt = optim.Adam(params=model.parameters(),lr=0.01)
criterion = nn.NLLLoss()

torch.autograd.set_detect_anomaly(True)

for i in range(10):
    print('Epoch:',i+1)
    
    opt.zero_grad()
    
    out_list, cov_loss = model(tensor_art, tensor_sum)
    
    loss = 0
    for j in range(out_list.shape[0]):
        loss += criterion(out_list[j],tensor_sum[j,1:])
        
    loss += cov_loss
    
    #loss = criterion(out_list,tensor_sum[:,1:])+cov_loss
    print('Loss:',loss)
    
    loss.backward()
    opt.step()
    

#model(tensor_art[0:3], tensor_sum[0:3])
