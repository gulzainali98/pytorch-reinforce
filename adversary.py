from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import math

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

def complete_video(sd, ntokens=1518, device=None, save_seq= None):
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
        try:
            return sd[:,0:ntokens,:]
        except Exception as e:
            print("return saved sequence")
            print(save_seq.shape)
            print(sd.shape)
            if save_seq.shape[1]==ntokens:
                return save_seq
            elif save_seq.shape[1]<ntokens:
                fea= np.load("./blank_features.npy",allow_pickle=True)
                # new_tensor= torch.zeros((sd.shape[0],sd.shape[1],50))
                # new_tensor= torch.zeros(sd.shape)
                # diff= sd.shape[0]
                fea= np.expand_dims(fea,axis=0)
                fea= np.expand_dims(fea,axis=0)
                fea= torch.tensor(fea).to(device)

                while save_seq.shape[1]!=ntokens:
                # sd= torch.cat([sd,fea],dim=0)
                    save_seq= torch.cat((save_seq,fea),dim=1)
                # print(sd.shape)
                return save_seq
            else:
                return save_seq[:,0:ntokens,:]
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


class TransformAdversary(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, device,dropout=0.5):
        super(TransformAdversary, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout).to(device)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout).to(device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers).to(device)
        self.encoder = nn.Embedding(ntoken, ninp).to(device)
        self.ninp = ninp
        # print(ntoken*ninp)
        # self.decoder = nn.Linear(ntoken*ninp, ntoken*ninp)
        self.decoder= nn.Conv1d(1518, 1518, 1, stride=1)
        self.device= device
        self.sigmoid= nn.Sigmoid()

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
        src= src.to(self.device)
        self.pos_encoder= self.pos_encoder.to(self.device)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        # print(output.shape)
        # output= output.view(inp.shape[0],-1)
        # output= output.unsqueeze(0)
        # print(output.shape)
        # print(output.shape)
        # print(output.shape)
        output = self.decoder(output)
        output= output.squeeze(0)
        output =self.sigmoid(output)
        output= output.reshape((src.shape[0],src.shape[1],1024))
        # output= output.unsqueeze(0)
        return output
# i am not passing positional encoding

def make_adversary(ntokens, device):
    ntokens = ntokens # the size of vocabulary
    emsize = 1024 # embedding dimension
    nhid = 2048 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    device=device
    model =  TransformAdversary(ntokens, emsize, nhead, nhid, nlayers, device,dropout)
    return model

def manipulate(m, a, train_dataset=None, train_keys=None, optimizer=None, scheduler=None, device=None, use_gpu=True, args=None, together=False, epoch=200, s_d=None):
    if not together:
        print("==> Start training Adversary")
    else:
        print("==> Start training Adversary and Model together")
    model= m
    adversary= a
    start_time = time.time()
    model.train()
    adversary.train()
    if not together:
        for param in model.parameters():
            param.requires_grad = False
    baselines = {key: 0. for key in train_keys} # baseline rewards for videos
    reward_writers = {key: [] for key in train_keys} # record reward changes for each video
    reward_writers_nll = {key: [] for key in train_keys}
    activation = nn.Softmax(dim=1)
    epis_reward_nll=[]
    # loss = nn.NLLLoss()
    loss= nn.CrossEntropyLoss()
    label= torch.full((1,), 1, dtype=torch.long).to(device)
    save_seq=None
    count=0
    # for epoch in range(start_epoch, args.max_epoch):
    for epoch in range(0, epoch):

        idxs = np.arange(len(train_keys))
        np.random.shuffle(idxs) # shuffle indices


        for idx in idxs:
            key = train_keys[idx]
            # seq = dataset[key]['features'][...] # sequence of features, (seq_len, dim)

            seq = train_dataset[key]['features'][...] # sequence of features, (seq_len, dim)

            seq = torch.from_numpy(seq).unsqueeze(0) # input shape (1, seq_len, dim)

            if use_gpu: seq = seq.cuda()
            length= seq.shape[1]
            seq_updated= complete_video(seq,device=device)

            src_mask=adversary.generate_square_subsequent_mask(seq.size(0))
            seq_manipulated= adversary(seq_updated,src_mask)
            seq_manipulated= seq_manipulated[:,:length,:]
            # print(seq_manipulated.shape)
            # print(seq.shape)
            probs = model(seq_manipulated) # output shape (1, seq_len, 1)


            cost = args.beta * (probs.mean() - 0.5)**2 # minimize summary length penalty term [Eq.11]
            m = Bernoulli(probs)
            epis_rewards = []
            for _ in range(args.num_episode):
                actions = m.sample()
                log_probs = m.log_prob(actions)

                if count==0:

                    pick_idxs = actions.squeeze().nonzero().squeeze()
                    save_seq=seq[:,pick_idxs,:]
                    count=1

                reward, nll = compute_reward(seq, actions,s_d, loss=loss, label=label, activation=activation ,use_gpu=use_gpu, device=device, save_seq=save_seq)
                # reward = compute_reward_old(seq, actions,use_gpu=use_gpu)

                expected_reward = log_probs.mean() * (reward - baselines[key])
                cost -= expected_reward # minimize negative expected reward
                epis_rewards.append(reward.item())
                # epis_reward_nll.append(nll.item())

            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards) # update baseline reward via moving average
            reward_writers[key].append(np.mean(epis_rewards))
            # reward_writers_nll[key].append(np.mean(epis_reward_nll))
        # if (epoch+1)%100==0:
        #     evaluate_save(model, dataset, test_keys, use_gpu, i=i)
        epoch_reward = np.mean([reward_writers[key][epoch] for key in train_keys])
        # epoch_nll_reward= np.mean([reward_writers_nll[key][epoch] for key in train_keys])
        print("Manipulation epoch {}/{}\t reward {}\t nll_reward {}\t".format(epoch+1, args.max_epoch, epoch_reward, 0))
    return model,adversary



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
