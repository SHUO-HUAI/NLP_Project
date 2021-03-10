import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from collections import Counter
from torch.autograd import Variable
import time
import classes

class Model(nn.Module):

  def __init__(self, dic,  emb_dim=128, hidden_dim=256):
    super(Model, self).__init__()

    self.emb_dim = emb_dim
    self.hidden_dim = hidden_dim
    self.word_count = len(dic.word2idx)

    self.embed = nn.Embedding(self.word_count, self.emb_dim)
    self.encoder = nn.LSTM(input_size=self.emb_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)


  def forward(self, inputs):

    print('Input shape:',inputs.shape)

    embeddings = self.embed(inputs)
    print('Embeddings shape:',embeddings.shape)

    encoded, _ = self.encoder(embeddings)
    print('Encoded shape:',encoded.shape)

    return

