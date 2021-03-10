import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from collections import Counter
from torch.autograd import Variable
import time
import classes
import functions
from functions import to_cuda

class Model(nn.Module):

  def __init__(self, dic,  emb_dim=128, hidden_dim=256):
    super(Model, self).__init__()

    self.emb_dim = emb_dim      # Word embedding dimension
    self.hidden_dim = hidden_dim    # Hidden layer for encoder (BiLSTM) and decoder (LSTM)
    self.word_count = len(dic.word2idx)     # Vocabulary size
    

    
    self.embed = nn.Embedding(self.word_count, self.emb_dim)
    self.encoder = nn.LSTM(input_size=self.emb_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
    


  def forward(self, inputs):

    print('Input shape:',inputs.shape)

    embeddings = self.embed(inputs)
    print('Embeddings shape:',embeddings.shape)

    encoded, _ = self.encoder(embeddings)
    print('Encoded shape:',encoded.shape)
    
    coverage = np.zeros(inputs.shape,np.long)   # Coverage has size (b x seq_length) like attention
    coverage = torch.Tensor(coverage)
    coverage = to_cuda(coverage)
    print('Coverage shape:',coverage.shape)

    return

