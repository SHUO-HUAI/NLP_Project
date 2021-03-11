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

  def __init__(self, dic, art_len, emb_dim=128, hidden_dim=256):
    super(Model, self).__init__()

    self.emb_dim = emb_dim      # Word embedding dimension
    self.hidden_dim = hidden_dim    # Hidden layer for encoder (BiLSTM) and decoder (LSTM)
    self.word_count = len(dic.word2idx)     # Vocabulary size
    self.art_len = art_len
    self.dictionary = dic   # Saving the dictionary

    
    self.embed = nn.Embedding(self.word_count, self.emb_dim)
    self.encoder = nn.LSTM(input_size=self.emb_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
    self.decoder = nn.LSTM(input_size=self.emb_dim, hidden_size=hidden_dim, batch_first=True)
    
    
    self.Wh = nn.Linear(self.hidden_dim*2, self.hidden_dim) # for obtaining e from encoder hidden states
    self.Ws = nn.Linear(self.hidden_dim, self.hidden_dim) # for obtaining e from current state
    self.Wc = nn.Linear(self.art_len, self.hidden_dim) # for obtaining e from context vector
    self.v = nn.Linear(hidden_dim, 1) # for changing to scalar


  def forward(self, inputs, target, Train=True):

    input_len = inputs.size(-1)    # max input sequence length
    target_len = target.size(-1)  # max target sequence length


    inputs = inputs.view(-1,input_len)    # If I pass only 1 article and 1 summary I add one dimension at the beginning (useful for test)
    target = target.view(-1,target_len)

    batch_size = inputs.size(0)

    inputs = to_cuda(inputs)
    target = to_cuda(target)

    print('Input shape:',inputs.shape)  # Size [b x input_len]

    embedded_inputs = self.embed(inputs)    # Size [b x input_len x emb_dim]
    print('Embeddings shape:',embedded_inputs.shape)

    encoded, _ = self.encoder(embedded_inputs)  # Size [b x input_len x 2*hidden_dim]
    print('Encoded shape:',encoded.shape)
    
    coverage = np.zeros(inputs.shape,np.long)   # Coverage has size (b x seq_length)
    coverage = torch.Tensor(coverage)
    coverage = to_cuda(coverage)
    print('Coverage shape:',coverage.shape)
    
    next_input = to_cuda(target[:,0])       # First word of summary (should be <SOS>)
    #print(self.dictionary.idx2word[next_input])
    
    out_list = []   # Output list
    
    for i in range(2):#target_len - 1):
    
        embedded_target = self.embed(next_input)    # size [b x emb_dim]
        
        # With unsqueeze size becomes [b x 1 x emb_dim]
        # 1 is the sequence length for the LSTM
        
        if i==0:
            state, C = self.decoder(embedded_target.unsqueeze(1))
        else:
            state, C = self.decoder(embedded_target.unsqueeze(1), C)
    
        
        # state: [b x 1 x hidden_dim]
        
        # ATTENTION
        # Contiguous creates a new tensor, preserving order
        # New tensor will have shape [b*inputs_len x hidden_dim*2]
        # So is very important to preserve order, since now batches are concatenated
        
        # Wh maps from hidden_dim*2 -> hidden_dim
        
        # attn1 shape [b*inputs_len x hidden_dim]
        attn1 = self.Wh(encoded.contiguous().view(-1,encoded.size(2))) + self.Ws(state.squeeze()).repeat(input_len,1) + self.Wc(coverage).repeat(input_len,1)
        
        attn2 = self.v(attn1) # Shape [b*input_len x 1]
        
        attn = F.softmax(attn2.view(batch_size,input_len), dim=1) # Shape [b x input_len]


    return

