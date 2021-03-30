import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from collections import Counter
from torch.autograd import Variable
import time
import classes
import functions
import preprocessing
from preprocessing import get_unked
from functions import to_cuda


class Model(nn.Module):

    def __init__(self, dic, art_len, emb_dim=128, hidden_dim=256, max_oovs=100):
        super(Model, self).__init__()

        self.emb_dim = emb_dim  # Word embedding dimension
        self.hidden_dim = hidden_dim  # Hidden layer for encoder (BiLSTM) and decoder (LSTM)
        self.word_count = len(dic.word2idx)  # Vocabulary size
        self.art_len = art_len
        self.dictionary = dic  # Saving the dictionary
        self.max_oovs = max_oovs

        self.embed = nn.Embedding(self.word_count, self.emb_dim)
        self.encoder = nn.LSTM(input_size=self.emb_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(input_size=self.emb_dim, hidden_size=hidden_dim, batch_first=True)

        self.Wh = nn.Linear(self.hidden_dim * 2, self.hidden_dim)  # for obtaining e from encoder hidden states
        self.Ws = nn.Linear(self.hidden_dim, self.hidden_dim)  # for obtaining e from current state
        self.Wc = nn.Linear(self.art_len, self.hidden_dim)  # for obtaining e from context vector
        self.v = nn.Linear(hidden_dim, 1)  # for changing to scalar

        self.wh = nn.Linear(hidden_dim * 2, 1)  # for changing context vector into a scalar
        self.ws = nn.Linear(hidden_dim, 1)  # for changing hidden state into a scalar
        self.wx = nn.Linear(emb_dim, 1)  # for changing input embedding into a scalar

        self.V1 = nn.Linear(hidden_dim * 3, hidden_dim * 3)
        self.V2 = nn.Linear(hidden_dim * 3, self.word_count)

    def forward(self, inputs, target, train=True):

        input_len = inputs.size(-1)  # max input sequence length
        target_len = target.size(-1)  # max target sequence length

        inputs = inputs.view(-1, input_len)
        # If I pass only 1 article and 1 summary I add one dimension at the
        # beginning (useful for test)
        target = target.view(-1, target_len)
        
        unked_inputs = get_unked(inputs, self.dictionary)

        batch_size = inputs.size(0)

        inputs = to_cuda(inputs)
        unked_inputs = to_cuda(unked_inputs)
        target = to_cuda(target)

        # print('Input shape:',inputs.shape)  # Size [b x input_len]

        embedded_inputs = self.embed(unked_inputs)  # Size [b x input_len x emb_dim]
        # print('Embeddings shape:',embedded_inputs.shape)

        encoded, _ = self.encoder(embedded_inputs)  # Size [b x input_len x 2*hidden_dim]
        # print('Encoded shape:',encoded.shape)

        coverage = np.zeros(inputs.shape, np.long)  # Coverage has size (b x seq_length)
        coverage = torch.Tensor(coverage)
        coverage = to_cuda(coverage)
        # print('Coverage shape:',coverage.shape)

        cov_loss = 0

        if train:
            next_input = to_cuda(target[:, 0])  # First word of summary (should be <SOS>)
        
        else:
            next_input = self.dictionary.word2idx['<SOS>']
        

        out_list = []  # Output list
x
        for i in range(target_len - 1):

            embedded_target = self.embed(next_input)  # size [b x emb_dim]

            # With unsqueeze size becomes [b x 1 x emb_dim]
            # 1 is the sequence length for the LSTM

            if i == 0:
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
            attn1 = self.Wh(encoded.contiguous().view(-1, encoded.size(2))) + self.Ws(state.clone().squeeze()).repeat(
                input_len, 1)

            attn2 = self.v(attn1)  # Shape [b*input_len x 1]

            attn = F.softmax(attn2.view(batch_size, input_len), dim=1)  # Shape [b x input_len]

            # CONTEXT VECTOR

            context2 = torch.bmm(attn.unsqueeze(1), encoded)  # [b x 1 x in_seq] * [b x in_seq x hidden*2]
            context = context2.squeeze()  # [b x hidden*2] One array of [hidden*2] for each article

            # PROBABILITY OF GENERATING (one for each article, at each time step)

            # Depends on context vector, state of decoder, and input of decoder (embedded_target)
            p_gen = torch.sigmoid(self.wh(context) + self.ws(state.squeeze()) + self.wx(embedded_target))  # [b]
    
            # Coverage loss (better pay attention on less covered words)
            cov_loss += torch.sum(torch.min(attn.clone(), coverage.clone()))

            coverage += attn

            # Probability of vocabulary
            # Concat context + state -> (hidden_dim*3]

            cat = torch.cat([state.squeeze().view(batch_size, -1), context.view(batch_size, -1)], 1)
            v2_out = self.V2(self.V1(cat))

            p_vocab = F.softmax(v2_out, dim=1)  # Shape [b x vocab]
            p_vocab = v2_out
            
            #print(attn.shape)
            
            p_expanded_vocab = torch.zeros([batch_size,self.word_count + self.max_oovs])
            p_expanded_vocab[:,:self.word_count] = p_vocab
            p_expanded_vocab += 1/self.word_count
            #print(p_expanded_vocab)
            
            # ... PROBABILITY OF COPYING HERE ...
            # ... PROBABILITY OF COPYING HERE ...
            
            p_copy = torch.tensor(1) - p_gen[0][0]  # p_gen has shape [1,1] so doing [0][0] we get the raw number
            p_oov = []
  
            
            for b in range(inputs.shape[0]):
                attn_idx = 0
                for word in inputs[b]:
                
                    p_expanded_vocab[b][word.item()] += p_copy*attn[b][attn_idx]
                        
                    attn_idx += 1
                    
            
            #p_vocab += 1/self.word_count
            p_expanded_vocab = F.softmax(p_expanded_vocab, dim=1)
    
            #print(p_oov)
            #time.sleep(10)
    
            # ... PROBABILITY OF COPYING HERE ...
            # ... PROBABILITY OF COPYING HERE ...
            
            
            out = p_expanded_vocab.max(1)[1]  # .squeeze()

            # out_list.append(out)
            out_list.append(p_expanded_vocab)
            #print(p_expanded_vocab.shape)
            #time.sleep(10)

            if train:
                next_input = to_cuda(target[:, i + 1])
                # print(self.dictionary.idx2word[next_input])

            else:
                next_input = to_cuda(out)

        # print('Out_List:', out_list)
        out_list = torch.stack(out_list, 1)
        # print('Out_List:', out_list)

        return out_list, cov_loss
