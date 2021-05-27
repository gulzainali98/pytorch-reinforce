from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import math
from torch.nn import functional as F

import os
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
from tabulate import tabulate
from transformer_discriminator import *
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli

from utils import Logger, read_json, write_json, save_checkpoint
from models import *
from transformer_discriminator import *

from rewards import compute_reward
from rewards import *
import vsum_tools



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel_classifier(nn.Module):

    def __init__(self, ntokens, ninp, nhead, nhid, nlayers, num_networks,device,dropout=0.5):
        super(TransformerModel_classifier, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout).to(device)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout).to(device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers).to(device)
        self.encoder = nn.Embedding(ntokens, ninp).to(device)
        self.ninp = ninp
        self.decoder = nn.Linear(ntokens*ninp, num_networks).to(device)
        self.device= device
        self.sigmoid= nn.Sigmoid().to(device)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        # src = self.encoder(src) * math.sqrt(self.ninp)
        inp=src
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output= output.view(inp.shape[0],-1)
        # print(output.shape)
        output = self.decoder(output)
        # output =self.sigmoid(output)
        return output

def complete_video(sd, ntokens=600, device=None):
    if sd.shape[1]==ntokens:
        return sd
    if sd.shape[1]<ntokens:
        # add more vectors
        fea= np.load("./blank_features.npy",allow_pickle=True)
        # new_tensor= torch.zeros((sd.shape[0],sd.shape[1],50))
        # new_tensor= torch.zeros(sd.shape)
        # diff= sd.shape[0]
        fea= np.expand_dims(fea,axis=0)
        fea= np.expand_dims(fea,axis=0)
        fea= torch.tensor(fea).to(device)
        # print(fea.shape)

        # print(fea.shape)
        while sd.shape[1]!=ntokens:
        # sd= torch.cat([sd,fea],dim=0)
            sd= torch.cat((sd,fea),dim=1)
        # print(sd.shape)
        return sd

    if sd.shape[1]>ntokens:
        return sd[:,0:ntokens,:]


class LSTMClassifier(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, num_networks,in_dim=1024, hid_dim=1024, num_layers=1, cell='lstm'):
        super(LSTMClassifier, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        self.num_networks=num_networks
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, num_networks)


    def forward(self, x):
        h, _ = self.rnn(x)
        # p = F.sigmoid(self.fc(h))

        h= h.view(1,-1)
        print(h.shape)
        p= self.fc(h)

        # p= self.activation(self.fc(h))
        return p
#

def print_save(txt, location="results.txt"):
    f = open(location,"a")
    f.write(str(txt)+" \n")
    f.close()

class Transformer_Holder:
    def __init__(self,ntokens, emsize, nhead, nhid, nlayers,num_networks, dropout=None):
        device= torch.device("cuda:0")
        self.model=TransformerModel_classifier(ntokens, emsize, nhead, nhid, nlayers, num_networks,device,dropout)
    def complete_video(self,sd, ntokens_video=600, device=None):
        device= torch.device("cuda:0")
        if sd.shape[1]==ntokens_video:
            return sd
        if sd.shape[1]<ntokens_video:
        # add more vectors
            fea= np.load("./blank_features.npy",allow_pickle=True)
            # new_tensor= torch.zeros((sd.shape[0],sd.shape[1],50))
            # new_tensor= torch.zeros(sd.shape)
            # diff= sd.shape[0]
            fea= np.expand_dims(fea,axis=0)
            fea= np.expand_dims(fea,axis=0)
            fea= torch.tensor(fea).to(device)
            # print(fea.shape)

            # print(fea.shape)
            while sd.shape[1]!=ntokens_video:
            # sd= torch.cat([sd,fea],dim=0)
                sd= torch.cat((sd.to(device),fea.to(device)),dim=1)
            # print(sd.shape)
            return sd

        if sd.shape[1]>ntokens_video:
            # print(ntokens_video)
            return sd[:,0:ntokens_video,:]
    def __call__(self,seq):
        src_mask= self.model.generate_square_subsequent_mask(seq.size(0))
        seq= self.complete_video(seq)
        return self.model(seq,src_mask)
    def to(self, device):
        self.model= self.model.to(device)
        return self



def make_classifier(num_networks):

    ntokens = 600# the size of vocabulary
    emsize = 1024 # embedding dimension
    nhid = 1024 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    print(dropout)
    model = Transformer_Holder(ntokens, emsize, nhead, nhid, nlayers,num_networks, dropout=0.2)
    # model =  LSTMClassifier(num_networks)
    return model

def learn_distribution(model_holder, train_keys, train_dataset, keys_map, args,use_gpu=True):
    model= model_holder.model
    optim= torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.stepsize > 0:
        scheduler= lr_scheduler.StepLR(optim, step_size=args.stepsize, gamma=args.gamma)
    model.train()
    loss= nn.CrossEntropyLoss()

    for epoch in range(60):
        epoch_cost=0
        for key in train_keys:
            seq = train_dataset[key]['features'][...] # sequence of features, (seq_len, dim)

            seq = torch.from_numpy(seq).unsqueeze(0) # input shape (1, seq_len, dim)
            src_mask = model.generate_square_subsequent_mask(seq.size(0))
            seq= model_holder.complete_video(seq)

            label= torch.tensor([keys_map[key]],dtype=torch.long).cuda()

            if use_gpu: seq = seq.cuda()

            output= model(seq,src_mask)
            # print(output.shape)
            cost= loss(output,label)
            optim.zero_grad()
            cost.backward()
            epoch_cost= epoch_cost+cost.item()
        print("Loss: {}".format(str(epoch_cost/159)))
    model.eval()
    m_return= model_holder
    m_return.model= model
    return m_return





if __name__=="__main__":

    ntokens = 50 # the size of vocabulary
    emsize = 1024 # embedding dimension
    nhid = 2048 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    # device=torch.device("cpu")
    # model =  TransformAdversary(ntokens, emsize, nhead, nhid, nlayers, device,dropout)
    #
    # src = torch.rand((10, 50, 1024))
    # tgt = torch.rand((10, 50, 1024))
    # src_mask = model.generate_square_subsequent_mask(src.size(0))
    # output = model(src, src_mask)
    # print(output.shape)
