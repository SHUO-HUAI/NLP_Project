import argparse
import os
from os import listdir
from os.path import isfile, join
from io import open
import torch
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
from preprocessing import read_files, prepare_data, zero_pad
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
sum_idx = prepare_data(summaries, dic)

padded_articles = zero_pad(art_idx)
padded_summaries = zero_pad(sum_idx)

print('Length of padded articles:', len(padded_articles[0]))
print('Length of padded summaries:', len(padded_summaries[0]))

tensor_art = torch.LongTensor(padded_articles)
tensor_sum = torch.LongTensor(padded_summaries)

model = Model(dic)
model(tensor_art)
