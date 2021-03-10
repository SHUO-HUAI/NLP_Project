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
import config

parser = argparse.ArgumentParser(description='PyTorch Get To The Point Training')
parser.add_argument('--path', type=str, default=config.path,
                    help='path to the training data')


args = parser.parse_args()

articles, summaries, dic = preprocessing.read_files(args.path)
word_count = len(dic)
print('Number of unique words:', word_count)
