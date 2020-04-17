import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dropout import RNNDropout, EmbeddingDropout, WeightDropout
from utils import repackage_hidden

class RNNModel(nn.Module):
    """
    Wrapper for Language Models.
    """
    def __init__(self, encoder, decoder, tie_weights=True, initrange=0.1):
        super(RNNModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.groups = ['encoder', 'rnn.0', 'rnn.1', 'rnn.2', 'decoder']
        
        if tie_weights:
            self.decoder.fc1.weight = self.encoder.embeddings.weight
            
        # Initialize parameters
        self.encoder.embeddings.weight.data.uniform_(-initrange, initrange)
        self.decoder.fc1.bias.data.zero_()
        self.decoder.fc1.weight.data.uniform_(-initrange, initrange)
        
    def reset_hidden(self):
        self.encoder.reset_hidden()
        
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        
    def unfreeze(self, ix):
        to_unfreeze = self.groups[ix:]
        for n, p in self.named_parameters():
            for group in to_unfreeze:
                if group in n: p.requires_grad = True
                    
    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        
    def forward(self, x, **kwargs):
        out = self.decoder(*self.encoder(x), **kwargs)
        return out
    
class RNNClassifier(nn.Module):
    """
    Wrapper for Classifier. Used for ULMFiT.
    """
    def __init__(self, encoder, decoder):
        super(RNNClassifier, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.groups = ['encoder', 'rnn.0', 'rnn.1', 'rnn.2', 'decoder']
        
    def reset_hidden(self):
        self.encoder.reset_hidden()
        
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        
    def unfreeze(self, ix):
        to_unfreeze = self.groups[ix:]
        for n, p in self.named_parameters():
            for group in to_unfreeze:
                if group in n: p.requires_grad = True
                    
    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        
    def forward(self, x):
        out, hidden, raw_out, dropped_out = self.encoder(x)
        logits = self.decoder(out, hidden[-1])
        return logits

class AWDLSTMEncoder(nn.Module):
    """
    AWD-LSTM Encoder as proposed by Merity et al. (2017) 
    """
    def __init__(self, vocab_sz, emb_dim, hidden_dim, num_layers=1, emb_dp=0.1, weight_dp=0.5, input_dp=0.3, hidden_dp=0.3, tie_weights=True, padding_idx=1):
        super(AWDLSTMEncoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_sz, emb_dim, padding_idx=padding_idx)
        self.emb_dp = EmbeddingDropout(self.embeddings, emb_dp)
        
        self.rnn = nn.ModuleList([nn.LSTM(emb_dim if l == 0 else hidden_dim, (hidden_dim if l != num_layers - 1 else emb_dim) if tie_weights else hidden_dim) for l in range(num_layers)])
        self.weight_dp = nn.ModuleList([WeightDropout(rnn, weight_dp) for rnn in self.rnn])
        self.hidden_dp = RNNDropout(hidden_dp)
        self.input_dp = RNNDropout(input_dp)

        self.hidden, self.cell = None, None
    
    def init_hidden(self, bs):
        weight = next(self.parameters())
        hidden = [weight.new_zeros(1, bs, self.rnn[i].hidden_size) for i in range(len(self.rnn))]
        cell  = [weight.new_zeros(1, bs, self.rnn[i].hidden_size) for i in range(len(self.rnn))]
        
        return hidden, cell
    
    def reset_hidden(self):
        self.hidden, self.cell = None, None
    
    def forward(self, x):
        msl, bs = x.shape

        # Initialize hidden states or detatch from history
        # We need to detatch or else the model will backprop
        # through previous batches on a new batch
        if self.hidden is None and self.cell is None:
            self.hidden, self.cell = self.init_hidden(bs)
        else:
            self.hidden = [repackage_hidden(h) for h in self.hidden]
            self.cell = [repackage_hidden(h) for h in self.cell]

        out = self.emb_dp(x)
        
        raw_output = []
        dropped_output = []
        out = self.input_dp(out)

        for i in range(len(self.rnn)):
            out, (self.hidden[i], self.cell[i]) = self.weight_dp[i](out, (self.hidden[i], self.cell[i]))
            raw_output.append(out)
            
            if i < len(self.rnn) - 1: 
                out = self.hidden_dp(out)
                dropped_output.append(out)

        # out is the final processed RNN output
        # self.hidden is a list of all hidden states. Use the last one.
        # raw_output is a list with all raw RNN outputs
        # dropped_output is a list with all RNN outputs with RNN dropout applied
        # dropped_output contains one less item than raw_output at all times
        return out, self.hidden, raw_output, dropped_output

class LSTMEncoder(nn.Module):
    """
    Basic LSTM Encoder for Language Modeling 
    """
    def __init__(self, vocab_sz, emb_dim, hidden_dim, num_layers=1, dropout=0.5):
        super(LSTMEncoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_sz, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.hidden, self.cell = None, None
    
    def init_hidden(self, bs):
        weight = next(self.parameters())
        nlayers = self.rnn.num_layers
        nhid = self.rnn.hidden_size
        
        return (weight.new_zeros(nlayers, bs, nhid), weight.new_zeros(nlayers, bs, nhid))
    
    def reset_hidden(self):
        self.hidden, self.cell = None, None
    
    def forward(self, x, lens=None):
        msl, bs = x.shape
        
        if self.hidden is None and self.cell is None:
            self.hidden, self.cell = self.init_hidden(bs)
        else:
            self.hidden, self.cell = repackage_hidden((self.hidden, self.cell))

        out = self.embeddings(x)
        out = self.dropout(out)

        out, (self.hidden, self.cell) = self.rnn(out, (self.hidden, self.cell))
        out = self.dropout(out)

        return out, self.hidden, self.cell

class DropoutLinearDecoder(nn.Module):
    """
    Linear Decoder with output RNN Dropout. Used with AWD LSTM. 
    """
    def __init__(self, net_arch_LDA, hidden_dim, vocab_sz, out_dp=0.4):
        super(DropoutLinearDecoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, vocab_sz)
        self.out_dp = RNNDropout(out_dp)
        # LDA IMPORT
        ac = net_arch_LDA
        self.net_arch = net_arch_LDA
         self.en1_fc     = nn.Linear(ac.num_input, ac.en1_units)             # 1995 -> 100 
        self.en2_fc     = nn.Linear(ac.en1_units, ac.en2_units)             # 100  -> 100 
        self.en2_drop   = nn.Dropout(0.2)
        self.mean_fc    = nn.Linear(ac.en2_units, ac.num_topic)             # 100  -> 50 
        self.mean_bn    = nn.BatchNorm1d(ac.num_topic)                      # bn for mean   LINE 20
        self.logvar_fc  = nn.Linear(ac.en2_units, ac.num_topic)             # 100  -> 50
        self.logvar_bn  = nn.BatchNorm1d(ac.num_topic)                      # bn for logvar 
        
        self.combo_fc   = nn.Linear(ac.num_topic + hidden_dim, ac.num_topic)  # SHOULD BE 50 -> 50??, 
        
        self.p_drop     = nn.Dropout(0.2)    
        # decoder
        self.decoder    = nn.Linear(ac.num_topic, ac.num_input)             # 50   -> 1995   #DECODER OF TOPIC MODEL
        self.decoder_bn = nn.BatchNorm1d(ac.num_input)                      # bn for decoder  
        # prior mean and variance as constant buffers                       
        prior_mean   = torch.Tensor(1, ac.num_topic).fill_(0)
        prior_var    = torch.Tensor(1, ac.num_topic).fill_(ac.variance)
        prior_logvar = prior_var.log()
        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar)
        # initialize decoder weight
        if ac.init_mult != 0:
            #std = 1. / math.sqrt( ac.init_mult * (ac.num_topic + ac.num_input))
            self.decoder.weight.data.uniform_(0, ac.init_mult)
        # remove BN's scale parameters
        self.logvar_bn .register_parameter('weight', None)
        self.mean_bn   .register_parameter('weight', None)
        self.decoder_bn.register_parameter('weight', None)
        self.decoder_bn.register_parameter('weight', None)
        #END IMPORT
        
    def forward(self, out, hidden, raw, dropped, return_states=False, compute_loss=True, avg_loss=True):
        # Applies RNN Dropout on the RNN output and 
        # appends to the dropped_output list. Raw_output 
        # and dropped_output should have equal number of 
        # elements now 
        out = self.out_dp(out)
        dropped.append(out)
        out = self.fc1(out)
        #LDA IMPORT
        #def forward(self, input, compute_loss=False, avg_loss=True):        
        # compute posterior
        h = tensor.ones                                                 # initialize h, with 0    
        en1 = F.softplus(self.en1_fc(out))                              # en1_fc   output   <--- should this be hidden? or another input?, match dimensions
        en2 = F.softplus(self.en2_fc(en1))                              # encoder2 output 
        en2 = self.en2_drop(en2)
        posterior_mean   = self.mean_bn  (self.mean_fc  (en2))          # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))          # posterior log variance
        posterior_var    = posterior_logvar.exp()                       
        # take sample
        eps = Variable(input.data.new().resize_as_(posterior_mean.data).normal_()) # noise
        z = posterior_mean + posterior_var.sqrt() * eps                 # reparameterization 
        p = F.softmax(z)                                                # mixture probability
        p = self.p_drop(p) 
        combo = torch.cat(p, hidden, 1)                             # maybe send it through another layer, add a 3rd encoder?, dimensions??
        combo = self.combo_fc(combo)                                    # check dimensions of this
        # do reconstruction
        recon = F.softmax(self.decoder_bn(self.decoder(combo)))         # reconstructed distribution over vocabulary  LINE 59 DECODER OF LANGUAGE MODEL
        loss = self.loss(out, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss) # what to give as input here? 
        #END IMPORT
        
        if return_states:
            return out, hidden, raw, dropped, recon, loss
        return out, recon, loss
    
        #IMPORT LDA LOSS
    def loss(self, input, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        NL  = -(input * (recon+1e-10).log()).sum(1)
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017, 
        # https://arxiv.org/pdf/1703.01488.pdf
        prior_mean   = Variable(self.prior_mean).expand_as(posterior_mean)
        prior_var    = Variable(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.net_arch.num_topic )
        # loss
        loss = (NL + KLD)
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean()
        else:
            return loss
        # END IMPORT
        
class LinearDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab_sz):
        super(LinearDecoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, vocab_sz)
        
    def forward(self, out, *args, **kwargs):
        return self.fc1(out)

class ConcatPoolingDecoder(nn.Module):
    """
    Concat Pooling Decoder from Howard & Ruder (2018)
    """
    def __init__(self, hidden_dim, bneck_dim, out_dim, dropout_pool=0.2, dropout_proj=0.1, include_hidden=True):
        super(ConcatPoolingDecoder, self).__init__()
        self.bn1 = nn.BatchNorm1d(hidden_dim * 3 if include_hidden else hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(bneck_dim)
        self.linear1 = nn.Linear(hidden_dim * 3 if include_hidden else hidden_dim * 2, bneck_dim)
        self.linear2 = nn.Linear(bneck_dim, out_dim)
        self.dropout_pool = nn.Dropout(dropout_pool)
        self.dropout_proj = nn.Dropout(dropout_proj)
        self.include_hidden = include_hidden
    
    def forward(self, out, hidden):
        _, bs, _ = out.shape
        avg_pool = F.adaptive_avg_pool1d(out.permute(1, 2, 0), 1).view(bs, -1)
        max_pool = F.adaptive_max_pool1d(out.permute(1, 2, 0), 1).view(bs, -1)
        if self.include_hidden:
            pooled = torch.cat([hidden[-1], avg_pool, max_pool], dim=1)
        else:
            pooled = torch.cat([avg_pool, max_pool], dim=1)
        out = self.dropout_pool(self.bn1(pooled))
        out = torch.relu(self.linear1(out))
        out = self.dropout_proj(self.bn2(out))
        out = self.linear2(out)
        return out
