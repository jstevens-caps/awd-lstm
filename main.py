import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from io import open
import hashlib
import argparse
from torch.utils.data import Dataset, DataLoader

from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from collections import defaultdict

#from transformers import WarmupLinearSchedule
from transformers import get_linear_schedule_with_warmup
from layers import RNNModel, AWDLSTMEncoder, DropoutLinearDecoder, LSTMEncoder, LinearDecoder
from utils import count_parameters, get_loaders, get_loaders_tok, drop_mult, extract_tags, _get_coherence, get_topics, get_word_list, create_batch, SortingTextDataLoader
from data import Corpus, Corpus_tok, Vocabulary, TextDataset
from predict import predict_model

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../data/wikitext-2', help='location of the data corpus')
parser.add_argument('--train', type=str, default='wiki.train.tokens', help='name of the training corpus')
parser.add_argument('--valid', type=str, default='wiki.valid.tokens', help='name of the validation corpus')
parser.add_argument('--test', type=str, default='wiki.test.tokens', help='name of the testing corpus')
parser.add_argument('--output', type=str, default='awd_lstm', help='output name')

parser.add_argument('--bs', type=int, default=80, help='batch size')
parser.add_argument('--eval_bs', type=int, default=80, help='evaluation batch size')
parser.add_argument('--bptt', type=int, default=80, help='bptt length')
parser.add_argument('--use_var_bptt', action='store_true', help='use variable length bptt')
parser.add_argument('--rebuild_dataset', action='store_true', help='force rebuild the dataset')
parser.add_argument('--load_vocab', action='store_true', help='load vocabulary')
parser.add_argument('--vocab_file', type=str, default='vocab.pth', help='pretrained vocabulary file')
parser.add_argument('--save_vocab', action='store_true', help='load vocabulary')

parser.add_argument('--encoder', type=str, default='awd_lstm', choices=['awd_lstm', 'lstm'], help='encoder')
parser.add_argument('--decoder', type=str, default='dropoutlinear', choices=['dropoutlinear', 'linear'], help='decoder')
parser.add_argument('--emb_dim', type=int, default=400, help='embedding dimensions')
parser.add_argument('--hidden_dim', type=int, default=1152, help='hidden dimensions')
parser.add_argument('--num_layers', type=int, default=3, help='number of rnn layers')
parser.add_argument('--emb_dp', type=float, default=0.2, help='embeddng dropout')
parser.add_argument('--hidden_dp', type=float, default=0.2, help='hidden to hidden dropout')
parser.add_argument('--input_dp', type=float, default=0.2, help='input dropout')
parser.add_argument('--weight_dp', type=float, default=0.2, help='dropconnect dropout')
parser.add_argument('--out_dp', type=float, default=0.2, help='output dropout')
parser.add_argument('--initrange', type=float, default=0.05, help='initialization range')
parser.add_argument('--tie_weights', action='store_true', help='tie embeddings and decoder weights')
parser.add_argument('--use_pretrained', action='store_true', help='use pretrained weights')
parser.add_argument('--freeze_encoder', action='store_true', help='freezes the encoder')
parser.add_argument('--pretrained_file', type=str, default='pretrained_wt103', help='pretrained model file')
parser.add_argument('--dm', type=float, default=1.0, help='dropout rate scaling')

parser.add_argument('--anneal_factor', type=float, default=4.0, help='learning rate anneal rate')
parser.add_argument('--lr', type=float, default=30, help='learning rate')
parser.add_argument('--no_lr_scaling', action='store_true', help='no lr scaling with var bptt or no auto anneal otherwise')
parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimzer to use')
parser.add_argument('--no_warmup', action='store_true', help='do not use linear warmups when using Adam')
parser.add_argument('--warmup_pct', type=float, default=0.1, help='percentage of steps for warmup')
parser.add_argument('--disc_rate', type=float, default=1.0, help='Discriminative learning rate scaling')

parser.add_argument('--epochs', type=int, default=2, help='epochs to train the network')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--alpha', type=float, default=2.0, help='AR alpha parameter')
parser.add_argument('--beta', type=float, default=1.0, help='TAR beta parameter')

parser.add_argument('--no_cuda', action='store_true', help='do not use CUDA')
parser.add_argument('--gpu', type=int, default=0, help='index of GPU to use')
parser.add_argument('--seed', type=int, default=42, help='random seed')

parser.add_argument('-f', '--en1-units',        type=int,   default=100)
parser.add_argument('-s', '--en2-units',        type=int,   default=100)
parser.add_argument('-t', '--num-topic',        type=int,   default=50)
# parser.add_argument('-b', '--batch-size',       type=int,   default=200)
# parser.add_argument('-o', '--optimizer',        type=str,   default='Adam')
# parser.add_argument('-r', '--learning-rate',    type=float, default=0.002)
parser.add_argument('-m', '--momentum',         type=float, default=0.99)
# parser.add_argument('-e', '--num-epoch',        type=int,   default=80)
parser.add_argument('-q', '--init-mult',        type=float, default=1.0)    # multiplier in initialization of decoder weight
parser.add_argument('-v', '--variance',         type=float, default=0.995)  # default variance in prior normal
parser.add_argument('--start',                  action='store_true')        # start training at invocation
parser.add_argument('--tokenized',  type=int, default=1, choices=[0,1], help='whether the input files are allready tokenized') 
parser.add_argument('--prior_train', action='store_true', help='whether we train the prior')
parser.add_argument('--kl_anneal', action='store_true', help='whether to anneal KL')
parser.add_argument('--margin', type=float, default=1, help='margin for KL anneal')

args = parser.parse_args()

if args.decoder == 'dropoutlinear': assert args.encoder == 'awd_lstm'

tag_ids = {'MISC':0, 'LOC':1, 'PER':2, 'ORG':3, 'O':4}
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>" 

# CUDA
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and not args.no_cuda else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed);
torch.cuda.manual_seed(args.seed);
torch.backends.cudnn.deterministic = True
print("Using device: {}".format(device))

if args.prior_train == True:
  print("Training Prior")


# Produce or load the dataset
path = args.path
fn = '{}/corpus.{}.data'.format(path, hashlib.md5(path.encode()).hexdigest())
if os.path.exists(fn) and not args.rebuild_dataset:
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    if args.load_vocab:
        print('Vocabulary has been loaded from {}'.format(args.vocab_file))
    if args.tokenized == 1:
        corpus = Corpus_tok(path, args.train, args.valid, args.test, load_vocab=args.load_vocab, vocab_file=args.vocab_file)
    else: 
        corpus = Corpus(path, args.train, args.valid, args.test, load_vocab=args.load_vocab, vocab_file=args.vocab_file)
    torch.save(corpus, fn)
    if args.save_vocab:
        with open('{}/{}'.format(path, args.vocab_file), 'wb') as f:
            torch.save([corpus.vocabulary.word2idx, corpus.vocabulary.idx2word], f)

vocab_sz = len(corpus.vocabulary)  

# Produce dataloaders
if args.tokenized == 1:
    print("Producing train dataloader...")
    train_loader = TextDataset(path, args.train, corpus.vocabulary)
    dlt = DataLoader(train_loader, batch_size=args.bs, drop_last=True)
    train_data = SortingTextDataLoader(dlt)
    print("Num sentences train loader:", len(train_loader))
    print("Producing val dataloader...")
    valid_loader = TextDataset(path, args.valid, train_loader.vocabulary)
    dlv = DataLoader(valid_loader, batch_size=args.bs, drop_last=True)
    valid_data = SortingTextDataLoader(dlv)
    print("Num sentences valid loader:", len(valid_loader))
    print("Producing test dataloader...")
    test_loader = TextDataset(path, args.test, valid_loader.vocabulary)
    dlte = DataLoader(test_loader, batch_size=args.bs, drop_last=True)
    test_data = SortingTextDataLoader(dlte)
    corpus.vocabulary = test_loader.vocabulary
    print("Num sentences test loader:", len(test_loader))
else:
    train_data = get_loaders(corpus.train, args.bs, args.bptt, use_var_bptt=args.use_var_bptt)
    valid_data = get_loaders(corpus.valid, args.eval_bs, args.bptt)
    test_data = get_loaders(corpus.test, args.eval_bs, args.bptt)

    # train_loader = get_loaders_tok(corpus.train, args.bs, args.bptt, corpus.vocabulary, tag_ids, device, word_dropout=0., use_var_bptt=args.use_var_bptt)
    # valid_loader = get_loaders_tok(corpus.valid, args.bs, args.bptt, corpus.vocabulary, tag_ids, device, word_dropout=0.)
    # test_loader  = get_loaders_tok(corpus.test, args.bs, args.bptt, corpus.vocabulary, tag_ids, device, word_dropout=0.)
    # print("train_loader", train_loader)
    #print("train_loader", train_loader[0])

# Prepare arguments as input for encoder
net_arch = args
print("Vocabulary Size:", vocab_sz)
net_arch.device_tn = device

# Construct encoder
if args.encoder == 'awd_lstm':
    encoder = AWDLSTMEncoder(net_arch, prior_train=args.prior_train, anneal_KL=args.kl_anneal, anneal_KL_margin=args.margin, vocab_sz=vocab_sz, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
                             num_layers=args.num_layers, emb_dp=args.emb_dp, weight_dp=args.weight_dp,
                             input_dp=args.input_dp, hidden_dp=args.hidden_dp, tie_weights=args.tie_weights)
elif args.encoder == 'lstm':
    encoder = LSTMEncoder(vocab_sz=vocab_sz, emb_dim=args.emb_dim, num_layers=args.num_layers,
                          hidden_dim=args.emb_dim if args.tie_weights else args.hidden_dim, dropout=args.weight_dp)

# Construct decoder
if args.decoder == 'dropoutlinear':
    decoder = DropoutLinearDecoder(net_arch, hidden_dim=args.emb_dim if args.tie_weights else args.hidden_dim,
                                   vocab_sz=vocab_sz, out_dp=args.out_dp)
elif args.decoder == 'linear':
    decoder = LinearDecoder(hidden_dim=args.emb_dim if args.tie_weights else args.hidden_dim, vocab_sz=vocab_sz)

# Produce model
model = RNNModel(encoder, decoder, tie_weights=args.tie_weights, initrange=args.initrange)
model = drop_mult(model, dm=args.dm)
if args.freeze_encoder:
    model.freeze()
    model.unfreeze(-1)
print(model)

# Pretrained
if args.use_pretrained:
    print("Using pretrained model {}".format(args.pretrained_file))
    with open('{}/{}'.format(path, args.pretrained_file), 'rb') as f:
        inc = model.load_state_dict(torch.load(f, map_location=device), strict=False)
    print(inc)

model = model.to(device);

# Parameter groups
p_groups = [{'name': '0', 'params': []}, {'name': '1', 'params': []}]
for n, p in model.named_parameters():
    if 'rnn' in n:
        p_groups[1]['params'].append(p)
    else:
        p_groups[0]['params'].append(p)

# Optimization setup
criterion = nn.CrossEntropyLoss()
optimizer, scheduler = None, None
if args.optimizer == 'sgd':
    optimizer = optim.SGD(p_groups, lr=args.lr)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(p_groups, lr=1e-3)
    steps = len(train_data) * args.epochs
    if not args.no_warmup:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(steps * args.warmup_pct), num_training_steps = steps)
        #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(steps * args.warmup_pct), t_total=steps)

print("Optimization settings:")
print(optimizer)
print("Scheduler: {}".format(scheduler))
print("The model has {:,} trainable parameters".format(count_parameters(model)))

# Training setup
best_loss = np.inf
best_epoch = 0
train_losses = []
valid_losses = []
best_components = None

# Training!
print("Beginning training")
try:
    num_batch = 0
    num_words = 0 
    for e in range(1, args.epochs + 1):
        model.train()
        model.reset_hidden()
        train_loss = 0
        train_KL = 0 
        with tqdm(total=len(train_data)) as t:
            for batch in train_data:
                #print("len sentences", sentences)
                if args.tokenized == 1:
                  sentences = batch[0]
                  labels = batch[1]
                  x, y, tags = create_batch(sentences, 
                                                      labels, 
                                                      corpus.vocabulary, 
                                                      tag_ids, 
                                                      device)
                else:
                  x, y = batch
                num_words += x.size(0) 
                num_batch_words = x.size(0)

                # Scale learning rate to sequence length
                # if args.use_var_bptt and not args.no_lr_scaling:
                #     seq_len, _ = x.shape
                #     optimizer.param_groups[0]['lr'] = args.lr * seq_len / args.bptt

                # # # Adjust discriminative learning rates
                # for i in range(len(optimizer.param_groups)):
                #     optimizer.param_groups[i]['lr'] /= args.disc_rate ** i

                x = x.to(device)
                y = y.to(device)

                out = model(x, return_states=True)
                if args.encoder == 'awd_lstm': out, hidden, raw_out, dropped_out, p, KL = out
                raw_loss = criterion(out.view(-1, vocab_sz), y)

                # AR/TAR
                loss = raw_loss #+ loss_LDA
                KL = KL.sum()
                #print("raw_loss_size", raw_loss.size())
                # if args.encoder == 'awd_lstm':
                #     # print("dropped_out.size", len(dropped_out)) 
                #     loss += args.alpha * dropped_out[-1].pow(2).mean()
                #     loss += args.beta * (raw_out[-1][1:] - raw_out[-1][:-1]).pow(2).mean()
                loss += KL # <-- I think add it here, because alpha & beta are part of criterion
                train_KL += KL
                # print("Train KL", KL.sum())
                optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                t.set_postfix({'lr{}'.format(i): optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))})
                if scheduler is not None: scheduler.step()

                # Restore original LR
                if args.use_var_bptt and not args.no_lr_scaling:
                    optimizer.param_groups[0]['lr'] = args.lr

                t.update()
                train_loss += loss.item()
                num_batch += 1
        
        train_loss /= num_words
        train_KL /= num_words
        train_losses.append(train_loss)

        model.eval() 
        model.reset_hidden()
        valid_loss = 0 
        KLD = 0 
        num_val_batch = 0
        num_val_words = 0
        for batch in tqdm(valid_data):
            with torch.no_grad():
                if args.tokenized == 1:
                  sentences = batch[0]
                  labels = batch[1]
                  x, y, tags = create_batch(sentences, 
                                                      labels, 
                                                      corpus.vocabulary, 
                                                      tag_ids, 
                                                      device)
                else:
                  x, y = batch
                                                      
                num_val_words += x.size(0) 
                num_batch_words = x.size(0)
                x = x.to(device)
                y = y.to(device)
                # print("size of x", x.size())
                # print("size of y", y.size())

                out = model(x)
                out, p, KL = out
                KL = KL.sum()
                # print("KL val:", KL.sum())
                # print("size of out", out.size())
                loss = criterion(out.view(-1, vocab_sz), y) + KL
                
                valid_loss += loss.item()
                KLD += KL / num_val_words 
                num_val_batch += 1
        
        valid_loss /= num_val_words
        valid_losses.append(valid_loss)

        # Track and anneal LR
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = e 
            print("Best loss so far. Saving model.") 
            with open('{}/{}.pth'.format(path, args.output), 'wb') as f:
                torch.save(model.state_dict(), f)
            #best_components = model.beta
        else:
            if not args.use_var_bptt and not args.no_lr_scaling:
                optimizer.param_groups[0]['lr'] /= args.anneal_factor
        cur_lr = optimizer.param_groups[0]['lr']

        print("Epoch {:3} | Train Loss {:.4f} | Train Ppl {:.4f} | Train KL {:.4f} | Valid Total Loss {:.4f} | Valid KL {:.4f} |Valid Ppl {:.4f} | LR {:.4f}".format(e, train_loss, np.exp(train_loss), train_KL, valid_loss, KLD, np.exp(valid_loss), cur_lr))
        num_batch = 0
        num_val_batch = 0
except KeyboardInterrupt:
    print("Exiting training early")

# Load best saved model
print("Loading best model")
with open('{}/{}.pth'.format(path, args.output), 'rb') as f:
    model.load_state_dict(torch.load(f))

# Testing evaluation
print("Evaluating model")
model.eval()
model.reset_hidden()
test_loss = 0
KLD = 0 
num_test_words = 0
total_labels = 0.
correct = 0
num_sentences = 0
test_pred = []
y_true = []
word_list = []
tag_list = []
id2tag = {v: k for k, v in tag_ids.items()}

for batch in tqdm(test_data):
  with torch.no_grad():
        if args.tokenized == 1:
          sentences = batch[0]
          labels = batch[1]
          x, y, tags = create_batch(sentences, 
                                  labels, 
                                  corpus.vocabulary, 
                                  tag_ids, 
                                  device) 
        else:
          x, y = batch
          tags = torch.ones_like(x)

        #sentences = [corpus.vocabulary.idx2word[idx] for idx in sentence_idxs]
        num_test_words += x.size(0) 
        num_batch_words = x.size(0) 
        x = x.to(device) 
        y = y.to(device) 

        out = model(x) 
        out, p, KL = out 
        KL = KL.sum() 
       
        loss = criterion(out.view(-1, vocab_sz), y) + KL
        KLD += KL

        test_loss += loss.item()
        # probs = torch.softmax(p, -1) # is allready softmaxed
        _, predicted = torch.max(p.data, -1) 
        predicted = predicted.squeeze(0)
        idx2word = corpus.vocabulary.idx2word
        for inst in x.transpose(0,1):
          #print("size x", inst.size())
          #words = extract_tags(inst, corpus.vocabulary.idx2word) 
          if len(inst) == 1:
            words = [idx2word[inst.item()]]
          else:
            words = [idx2word[q.item()] for q in inst]
          word_list.append(words)


        for p in predicted.transpose(0,1):
          if len(p) == 1:
            tag_pred = p.item() 
          else: 
            tag_pred = [q.item() for q in p]
          #tag_pred = extract_tags(p, id2tag)
          tag_list.append(tag_pred)

        #l = labels.tolist()
        for t in tags:
          if len(t) == 1:
            tag = id2tag[t].item()
          else: 
            tag = [id2tag[q.item()] for q in t]
          y_true.append(tag)

        # for l in labels:
        #   tag_true = extract_tags(l, id2tag)
        #   y_true.append(tag_true)

test_loss /= num_test_words
KLD /= num_test_words 

F1_full = 0
co_score = 0
if args.tokenized == 0:
  print("Test Total Loss {:.4f} | Test KL {:.4f} | Test Ppl {:.4f} | Test F1 {:.4f} | Test TCHR {:.4f} ".format(test_loss, KLD, np.exp(test_loss), F1_full, co_score))

# print("instance of y_true", )
# print("instance of test pred", test_pred[0])
#print("len sent, len pred", len(test_pred[0][0]), len(test_pred[0][1]))

#print("len word, tag, true, list", len(word_list), len(tag_list), len(y_true))
for i in range(len(word_list)):
  test_pred.append((word_list[i], tag_list[i], y_true[i]))

# extract predictions from tuple (sentence, tags)
y_pred = [j for _,j,_ in test_pred] 
count = 0
for i in y_true:
  count += len(i)

stop_words = ['de', 'en', 'van', 'ik', 'te', 'dat', 'die', 'in', 'een', 'hij', 'het', 'niet', 'zijn', 'is', 
              'was', 'op', 'aan', 'met', 'als', 'voor', 'had', 'er', 'maar', 'om', 'hem', 'dan', 'zou', 'of', 
              'wat', 'mijn', 'men', 'dit', 'zo', 'door', 'over', 'ze', 'zich', 'bij', 'ook', 'tot', 'je', 'mij', 
              'uit', 'der', 'daar', 'haar', 'naar', 'heb', 'hoe', 'heeft', 'hebben', 'deze', 'u', 'want', 'nog', 
              'zal', 'me', 'zij', 'nu', 'ge', 'geen', 'omdat', 'iets', 'worden', 'toch', 'al', 'waren', 'veel', 
              'meer', 'doen', 'toen', 'moet', 'ben', 'zonder', 'kan', 'hun', 'dus', 'alles', 'onder', 'ja', 'eens', 
              'hier', 'wie', 'werd', 'altijd', 'doch', 'wordt', 'lezen', 'kunnen', 'ons', 'zelf', 'tegen', 'na', 'reeds', 
              'wil', 'kon', 'niets', 'uw', 'iemand', 'geweest', 'andere', '-DOCSTART-', EOS_TOKEN, SOS_TOKEN, PAD_TOKEN]

k=int(20) # Words to sample for each topic
print("\n Compiling top {:.0f} words...".format(k))
grand_total = [[]                                           for topic in range(args.num_topic)]
props_list  = [{'MISC':0, 'LOC':0, 'PER':0, 'ORG':0, 'O':0} for topic in range(args.num_topic)]
for inst in test_pred:
  sen_len = len(inst[0])
  for i in range(sen_len):
    word = inst[0][i]
    tag = inst[1][i]
    true = inst[2][i]
    if len(word) > 1 and not word in stop_words:
      grand_total[tag].append(word) # add word to list in index of its tag
      #print("propos list", tag, props_list[tag])
      props_list[tag][true] += 1


grand_total_lens = [len(i) for i in grand_total]   
for i in range(args.num_topic):   
  for key in props_list[i]:   
    props_list[i][key] /= grand_total_lens[i]   
print(props_list)  

# [('MISC', 0.0231), ('LOC', 0.0119), ('PER', 0.0276), ('ORG', 0.0208), ('O', 0.9165)]
print("Numer of tokens per cluster", grand_total_lens)
summo = sum(grand_total_lens)
lens_sum = [i/summo for i in grand_total_lens]
lens_sum = [ '%.3f' % elem for elem in lens_sum ]
print("Proportions of tokens clustered into each cluster:", lens_sum)

topwords = []
for i in range(args.num_topic):
  topwords.append(np.random.choice(grand_total[i], k))
  print("Top {:.0f} words of cluster indexed {:.0f}: {}".format(k, i, topwords[i]))
  print("Proportions of labels present in cluster indexed {}: {}".format(i, props_list[i]))

# def classific_report(y_true, y_pred): 
#   if len(y_pred) == 1 or len(y_true) == 1: 
#     return "Cant classify, only one element" 
#   else: return classification_report(y_true, y_pred) 

# Remove EOS tokens from y_pred and word_list:
# for instance in range(len(word_list)):
#   indices = [i for i, x in enumerate(word_list[instance]) if x == EOS_TOKEN]
#   y_pred[instance] = [i for j, i in enumerate(y_pred[instance]) if j not in indices]
#   word_list[instance] = [i for j, i in enumerate(word_list[instance]) if j not in indices]

# Eval full dataset on F1, precision and recall
# print("\nEvaluating on full test dataset")
# print(classific_report(y_true, y_pred))
# F1_full = 0
# F1_full = f1_score(y_true, y_pred)
# print('F1 full: %.3f '%F1_full)

# # eval on one instance 
# print("\nEvaluating on instance 0")
# print("len predicted", len(y_pred[0]))
# print("len true     ", len(y_true[0]))
# print("len words    ", len(word_list[0]))
# print("Predicted:", y_pred[0])
# print("True     :", y_true[0])
# print("Words    :", word_list[0])
# print(classific_report(y_true[0], y_pred[0]))
# print('F1 0: %.3f '%f1_score(y_true[0], y_pred[0]))

# print("\nEvaluating on instance 1")
# print("Predicted:", y_pred[1])
# print("True     :", y_true[1])
# print("Words    :", word_list[1])
# print(classific_report(y_true[1], y_pred[1]))
# print('F1 1: %.3f '%f1_score(y_true[1], y_pred[1]))

# print("test_pred", test_pred[0])
# k=int(10) # Words to sample for each topic
# print("\n Compiling top {:.4f} words...".format(k))
# misc_words, tuple_list = get_word_list(test_pred, 'MISC', rebuild=True)
# per_words, _           = get_word_list(tuple_list, 'PER')
# org_words, _           = get_word_list(tuple_list, 'ORG')
# loc_words, _           = get_word_list(tuple_list, 'LOC')
# misc_words_sample = np.random.choice(misc_words, k).tolist()
# per_words_sample = np.random.choice(per_words, k).tolist()
# org_words_sample = np.random.choice(org_words, k).tolist()
# loc_words_sample = np.random.choice(loc_words, k).tolist()
# print("Top MISC words: ", misc_words_sample)
# print("Top PER words:  ", per_words_sample)
# print("Top ORG words:  ", org_words_sample)
# print("Top LOC words:  ", loc_words_sample)

# n_components = args.num_topic - 1 # in estebandito (I think) it was the amount of topics, we subtract 1 because of 'O'
# #topics = get_topics(best_components, n_components, corpus.vocabulary.idx2word, k=k)

# print("Estebandito get_topics: topics")

#co_score = _get_coherence(best_components, n_components, corpus.vocabulary.idx2word, k=k, topics=args.num_topic)
print("Mean Topic Coherence: ", co_score)

print("Test Total Loss {:.4f} | Test KL {:.4f} | Test Ppl {:.4f} | Test F1 {:.4f} | Test TCHR {:.4f} ".format(test_loss, KLD, np.exp(test_loss), F1_full, co_score))

# Saving graphs
print("Saving loss data")
pd.DataFrame(data={'train': train_losses, 'valid': valid_losses}).to_csv('{}/{}.csv'.format(path, args.output), index=False)
with open('{}/{}.txt'.format(path, args.output), 'w') as f:
    f.write("Best loss {:.4f} | Best ppl {:.4f} | Epoch {} | Test loss {:.4f} | Test KL {:.4f} | Test Ppl {:.4f} | Test F1 {:.4f} | Test TCHR {:.4f} ".format(best_loss, np.exp(best_loss), best_epoch, test_loss, KLD, np.exp(test_loss), F1_full, co_score))
    f.write("Numer of tokens per cluster".format(grand_total_lens))
    f.write("Proportions of tokens clustered into each cluster: {}".format(lens_sum))
    for i in range(args.num_topic):
      f.write("Top {:.0f} words of cluster indexed {:.0f}: {}".format(k, i, topwords[i]))
      f.write("Proportions of labels present in cluster indexed {}: {}".format(i, props_list[i]))
    
    
