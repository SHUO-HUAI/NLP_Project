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
from classes import Dictionary
import preprocessing
import model
import functions
from functions import to_cuda
from preprocessing import read_files, prepare_data, prepare_summary, zero_pad, remove_pad, get_unked, prepare_art_sum, prepare_dictionary
from model import Model
import pickle
import config

parser = argparse.ArgumentParser(description='PyTorch Get To The Point Training')
parser.add_argument('--path', type=str, default=config.stories_path,
                    help='path to the training data')
parser.add_argument('--token_path', type=str, default=config.tokenized_path,
                    help='path to the training data')

args = parser.parse_args()

path = args.path
token_path = args.token_path

articles, summaries, dic = read_files(path, token_path)
# for i in dic.word2idx.keys():
#    print(i, dic.word2idx[i])
# exit()
word_count = len(dic)
print('Number of unique words:', word_count)

art_idx = prepare_data(articles, dic)
sum_idx = prepare_summary(summaries, dic)

# hello = prepare_data(['my name is pasquale'], dic)
# unked_hello = get_unked(hello, dic)
# print(hello)
# print(unked_hello)
# exit()

#prepare TRAIN
train_path = 'train_all.txt'
valid_path = 'val_all.txt'
test_path = 'test_all.txt'
dic_path = 'dictionary'
out_path = 'data_finish/'

#prepare_dictionary(train_path, dic_path)

dic = Dictionary()

with open(dic_path, 'rb') as input:
    dic = pickle.load(input)

print(len(dic.word2idx))

#exit()

prepare_art_sum(train_path, out_path+'train_set', dic)
    
with open(out_path+'train_set', 'rb') as input:
    padded_train = pickle.load(input)


prepare_art_sum(valid_path, out_path+'valid_set', dic)

with open(out_path+'valid_set', 'rb') as input:
    padded_valid = pickle.load(input)
    

prepare_art_sum(test_path, out_path+'test_set', dic)
    
with open(out_path+'test_set', 'rb') as input:
    padded_test = pickle.load(input)
    

print('Train size:',len(padded_train))
print('Valid size:',len(padded_valid))
print('Test size:',len(padded_test))

padded_articles = padded_train[:,0]
padded_sums = padded_train[:,1]

padded_articles = np.array([np.array(tmp) for tmp in padded_articles])
padded_sums = np.array([np.array(tmp) for tmp in padded_sums])

tensor_art = torch.LongTensor(padded_articles)
tensor_sum = torch.LongTensor(padded_sums)

articles_len = len(tensor_art[0])

model = Model(dic, articles_len)
model = to_cuda(model)

# TEMPORARY CODE
# TEMPORARY CODE
# TEMPORARY CODE

opt = optim.Adam(params=model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

torch.autograd.set_detect_anomaly(True)

for i in range(100):
    print('Epoch:', i + 1)

    opt.zero_grad()
    out_list, cov_loss = model(tensor_art[0:2], tensor_sum[0:2])
    # print(len(out_list[0][0]))
    loss = torch.tensor(0.)
    loss = to_cuda(loss)
    for j in range(out_list.shape[0]):
        # loss += criterion(out_list[j],tensor_sum[j,1:]) # '1:' Remove <SOS>

        k = remove_pad(tensor_sum[j, :])

        loss += criterion(torch.log(out_list[j, :k]), tensor_sum[j, :k])

        # loss += cov_loss

    # PRINT
    k = remove_pad(tensor_sum[0,:])
    #print(tensor_sum[0,:k])
    
    out_string = []
    for word in tensor_sum[0,:k]:
        out_string.append(dic.idx2word[word])
        
    #print(len(out_string))

    
    out_string = []
    for word in out_list[0, :k]:
        out_string.append(dic.idx2word[torch.argmax(word)])
        
    
    print(out_string)
    
    # PRINT

    # loss = criterion(out_list,tensor_sum[:,1:])+cov_loss
    print('Loss:', loss)

    loss.backward()
    opt.step()

# model(tensor_art[0:3], tensor_sum[0:3])
