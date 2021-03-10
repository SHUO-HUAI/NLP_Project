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

path = './data'

articles, summaries, dic = preprocessing.read_files(path)
word_count = len(dic)
print('Number of unique words:', word_count)
