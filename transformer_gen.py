import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from tensorboardX import SummaryWriter
import math


ntokens = 50 # the size of vocabulary
emsize = 1024 # embedding dimension
nhid = 1024 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value


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


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, device,dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout).to(device)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout).to(device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers).to(device)
        self.encoder = nn.Embedding(ntoken, ninp).to(device)
        self.ninp = ninp
        self.decoder = nn.Linear(ntokens*ninp, 1).to(device)
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
        output =self.sigmoid(output)
        return output
