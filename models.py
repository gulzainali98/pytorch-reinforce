import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np

__all__ = ['DSN','ensemble_DSN', 'base_DSN']

ntokens = 50 # the size of vocabulary
emsize = 1024 # embedding dimension
nhid = 2024 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value


class DSN(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        p = F.sigmoid(self.fc(h))
        return p


class base_DSN(nn.Module):
    """Deep Summarization Network"""
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(base_DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim*2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        f= self.fc(h)
        p = F.sigmoid(f)
        return p



class ensemble_DSN:
    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm',num_networks=3):
        print("doing t")
        self.num_networks=num_networks
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        self.arr=[]
        self.num_networks=num_networks
        for i in range(num_networks):
            if cell == 'lstm':
                self.arr.append(base_DSN(cell='lstm'))
            else:
                self.arr.append(base_DSN(cell='gru'))
    def eval(self):
        for i in range(self.num_networks):
            self.arr[i].eval()
    def train(self, mode=True):
        print("custom train")
        for i in range(self.num_networks):
            self.arr[i].train()
    def to(self, device):
        for i in range(self.num_networks):
            self.arr[i]=self.arr[i].to(device)
        return self
    def __call__(self, x, num_network, eval=False, eval_network=None):
        if eval_network is not None:
            network= self.arr[eval_network]

            # h, _ = self.arr(i)(x[i])
            p = network(x)

            return p
        elif eval:
            p=torch.zeros((x.shape[0],x.shape[1],1))
            for network in range(len(self.arr)):
                p= self.arr[network](x)
            p= p/self.num_networks
            return p

        else:
            network= self.arr[num_network]

            # h, _ = self.arr(i)(x[i])
            p= network(x)

            return p
#
#
# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)
#
#
# class TransformerModel(nn.Module):
#
#     def __init__(self, ntoken, ninp, nhead, nhid, nlayers, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),dropout=0.5):
#         super(TransformerModel, self).__init__()
#         from torch.nn import TransformerEncoder, TransformerEncoderLayer
#         self.model_type = 'Transformer'
#         self.pos_encoder = PositionalEncoding(ninp, dropout).to(device)
#         encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout).to(device)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers).to(device)
#         self.encoder = nn.Embedding(ntoken, ninp).to(device)
#         self.ninp = ninp
#         self.decoder = nn.Linear(ntokens*ninp, 2).to(device)
#         # self.decoder = nn.Linear(1024, 1).to(device)
#
#         self.device= device
#         self.sigmoid= nn.Sigmoid().to(device)
#
#         self.init_weights()
#     def complete_summary(self,sd):
#         ntokens=350
#         if sd.shape[1]==ntokens:
#             return sd
#         if sd.shape[1]<ntokens:
#             # add more vectors
#             fea= torch.FloatTensor(np.load("./blank_features.npy",allow_pickle=True)).to(self.device)
#             new_tensor= torch.zeros((sd.shape[0],ntokens,sd.shape[2]))
#             diff= sd.shape[1]
#             fea= fea.unsqueeze(0).unsqueeze(0)
#             print(fea.shape)
#             print(sd.shape)
#             while sd.shape[1]!=ntokens:
#                 sd= torch.cat([sd,fea],dim=1)
#             return sd
#
#         if sd.shape[1]>ntokens:
#             return sd[:,0:ntokens,:]
#
#     def generate_square_subsequent_mask(self, sz):
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask.to(self.device)
#
#     def init_weights(self):
#         initrange = 0.1
#         self.encoder.weight.data.uniform_(-initrange, initrange)
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)
#
#     def forward(self, src):
#         # src = self.encoder(src) * math.sqrt(self.ninp)
#         src_mask= self.generate_square_subsequent_mask(src.size(0))
#         src= self.complete_summary(src)
#         print(src.shape)
#         inp=src
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src, src_mask)
#         # output= output.view(inp.shape[0],-1)
#         # print(output.shape)
#         print(output.shape)
#         output = self.decoder(output)
#         output =self.sigmoid(output)
#         return output

if __name__=="__main__":
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout=dropout)

    src = torch.rand((1, 300, 1024))
    tgt = torch.rand((20, 50, 1024))
    src_mask = model.generate_square_subsequent_mask(src.size(0))
    output = model(src)
    print(output.shape)
