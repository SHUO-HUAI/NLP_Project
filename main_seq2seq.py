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
from preprocessing import read_files, prepare_data, prepare_summary, zero_pad, remove_pad
from model_all import Model
import config

parser = argparse.ArgumentParser(description='PyTorch Get To The Point Training')
# parser.add_argument('--path', type=list, default=None,
#                     help='path to the training data')
parser.add_argument('--split_file', type=dict, default=config.split_file, help='path to the training data')


def data_process(args):
    split_file = args.split_file

    test_files_name = split_file['test']
    train_files_name = split_file['train']
    val_files_name = split_file['val']

    # read all

    articles, summaries, dic = read_files(token_path)

    word_count = len(dic)
    print('Number of unique words:', word_count)

    art_idx = prepare_data(articles, dic)
    sum_idx = prepare_summary(summaries, dic)

    padded_articles = zero_pad(art_idx)
    padded_summaries = zero_pad(sum_idx)

    print('Length of padded articles:', len(padded_articles[0]))
    print('Length of padded summaries:', len(padded_summaries[0]))
    # Save function here


def main():
    args = parser.parse_args()


start = time.time()

tensor_art = torch.LongTensor(padded_articles)
tensor_sum = torch.LongTensor(padded_summaries)

articles_len = len(tensor_art[0])
print(time.time() - start)
exit()
model = Model(dic, articles_len)
model = to_cuda(model)

# TEMPORARY CODE
# TEMPORARY CODE
# TEMPORARY CODE

opt = optim.Adam(params=model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# torch.autograd.set_detect_anomaly(True)

for i in range(100):
    print('Epoch:', i + 1)

    opt.zero_grad()

    out_list, cov_loss = model(tensor_art, tensor_sum)

    # out_list_tmp

    loss = torch.tensor(0.)
    loss = to_cuda(loss)

    # loss_tmp = torch.tensor(0.)
    # loss_tmp = to_cuda(loss_tmp)
    for j in range(out_list.shape[0]):
        # loss += criterion(out_list[j],tensor_sum[j,1:]) # '1:' Remove <SOS>

        k = remove_pad(tensor_sum[j, 1:])
        # out_list_tmp

        loss += criterion(torch.log(out_list[j, :k - 1]), tensor_sum[j, 1:k])
        # out_list[j]
    loss += cov_loss

    # PRINT
    out_string = []
    for word in out_list[j, :k - 1]:
        out_string.append(dic.idx2word[torch.argmax(word)])

    print(out_string)
    # PRINT

    # loss = criterion(out_list,tensor_sum[:,1:])+cov_loss
    print('Loss:', loss)

    loss.backward()
    opt.step()

# model(tensor_art[0:3], tensor_sum[0:3])
