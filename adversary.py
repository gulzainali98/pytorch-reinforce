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


class LSTMAdversary(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, in_dim=1024, hid_dim=1024, num_layers=1, cell='lstm'):
        super(LSTMAdversary, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, 1024)
        self.activation= nn.Tanh()

    def forward(self, x):
        h, _ = self.rnn(x)
        # p = F.sigmoid(self.fc(h))
        p= self.activation(self.fc(h))
        return p
#
class forget(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1):
        super(forget, self).__init__()
        self.forget_gate=nn.Linear(in_dim, out_dim)

    def forward(self,x):
        return F.sigmoid(self.forget_gate(x))

def print_save(txt, location="results.txt"):
    f = open(location,"a")
    f.write(str(txt)+" \n")
    f.close()

def make_adversary(ntokens, device):
    ntokens = ntokens # the size of vocabulary
    emsize = 1024 # embedding dimension
    nhid = 2048 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    device=device
    model =  LSTMAdversary()
    return model
def get_f_mean(model, dataset, test_keys, use_gpu, adversary=None, forget=None,args=None):
    with torch.no_grad():
        model.eval()
        fms = []
        eval_metric = 'avg' if args.metric == 'tvsum' else 'max'
        # eval_metric='max'

        if args.verbose: table = [["No.", "Video", "F-score"]]

        if args.save_results:
            h5_res = h5py.File(osp.join(args.save_dir, 'result.h5'), 'w')

        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            if use_gpu: seq = seq.cuda()
            noise= adversary(seq)
            gen_labels=forget(seq)
            seq_manipulated= seq+(noise*Bernoulli(gen_labels).sample())
            probs = model(seq_manipulated)
            probs = probs.data.cpu().squeeze().numpy()

            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...]

            machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
            fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            if args.verbose:
                table.append([key_idx+1, key, "{:.1%}".format(fm)])

            if args.save_results:
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
                h5_res.create_dataset(key + '/fm', data=fm)


    if args.save_results: h5_res.close()

    mean_fm = np.mean(fms)
    model.train()
    return mean_fm

def manipulate(m, a, train_dataset=None,test_dataset=None ,train_keys=None,test_keys=None, optimizer=None,
                scheduler=None, device=None, use_gpu=True, args=None,
                together=False, reward_dict=None, epoch=120, s_d=None, arguments=None):

    if not together:
        print("==> Start training Adversary")
    else:
        print("==> Start training Adversary and Model together")
    model= m
    adversary= a
    f= forget()
    f= f.to(device)
    start_time = time.time()
    model.train()
    adversary.train()
    f.train()
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
    loss_forget= nn.CrossEntropyLoss()
    label= torch.full((1,), 1, dtype=torch.long).to(device)
    save_seq=None
    count=0
    max_mean=0
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
            # length= seq.shape[1]
            # seq_updated= complete_video(seq,device=device)

            # src_mask=adversary.generate_square_subsequent_mask(seq.size(0))
            noise= adversary(seq) #(1,seq_len,1024)
            # print(seq_manipulated.shape)
            # seq_manipulated= seq_manipulated[:,:length,:]
            # print(seq_manipulated.shape)
            # print(seq.shape)
            gen_labels=f(seq).to(device)
            seq_manipulated= seq+(noise*Bernoulli(gen_labels).sample())
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

                reward, nll = compute_reward(seq_manipulated, actions,s_d, loss=loss, label=label, activation=activation ,use_gpu=use_gpu, device=device, save_seq=save_seq)
                expectation = log_probs.mean().detach() * (reward.detach() - baselines[key])
                try:
                    r= reward_dict[key]
                    if r<expectation:
                        reward_dict[key]=expectation
                        label_forget=torch.full((probs.shape[0],1), 1, dtype=torch.long).to(device)
                    else:
                        label_forget=torch.full((probs.shape[0],1), 0, dtype=torch.long).to(device)
                except Exception as e:
                    reward_dict[key]=expectation

                    label_forget= torch.full((probs.shape[0],1), 1, dtype=torch.long).to(device)

                # partial impartial supervision

                # reward = compute_reward_old(seq, actions,use_gpu=use_gpu)
                reward= reward+ loss_forget(gen_labels,label_forget)
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
        mean=get_f_mean(model, test_dataset, test_keys, use_gpu, adversary=adversary,forget=f,args=arguments)
        if mean > max_mean:
            max_mean=mean
        # epoch_nll_reward= np.mean([reward_writers_nll[key][epoch] for key in train_keys])
        print("Manipulation epoch {}/{}\t reward {}\t nll_reward {}\t".format(epoch+1, args.max_epoch, epoch_reward, 0))
    print_save("="*20)
    print_save("Manipulation Max Mean: "+str(max_mean))
    return model,adversary, f



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
