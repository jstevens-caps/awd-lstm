import torch
import torch.utils.data as data_utils
import numpy as np
from tqdm import tqdm

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

def batchify_tup(data, bsz):
    # this function lets us batchify the tuple (sentences, labels)
    sentences, labels = data
    sen_batch = batchify(sentences, bsz)
    lab_batch = batchify(labels, bsz)
    return sen_batch, lab_batch

def create_batch(data, vocab, tag2id, device, word_dropout=0.):
    """
    Converts a list of sentences to a padded batch of word ids. Returns
    an input batch, output tags, a sequence mask over
    the input batch, and a tensor containing the sequence length of each
    batch element.
    :param sentences: a list of sentences, each a list of token ids
    :param labels: a list of outputs
    :param vocab: a Vocabulary object for this dataset
    :param device: 
    :param word_dropout: rate at which we omit words from the context (input)
    Notice we do not use word dropout (forget words from a vocabulary) but sequence dropout (forget words in a sentence)
    :returns: a batch of padded inputs, a batch of outputs, mask, lengths
    """
    # we get the max lenght sentence in a mini-batch to create the padding
    senntences, labels = data 
    # convert these tensors into lists
    sentences.to_list()
    labels.to_list()

    tok = np.array([([SOS_TOKEN] + sen.split() + [EOS_TOKEN]) for sen in sentences])
    seq_lengths = [len(sen)-1 for sen in tok] 
    max_len = max(seq_lengths) 
    pad_id = vocab[PAD_TOKEN]  #pad token 
     # padding of the sentences that a sorther than the max with 0
    pad_id_input = [
        [vocab[sen[t]] if t < seq_lengths[idx] else pad_id for t in range(max_len)]
            for idx, sen in enumerate(tok)]
    # we do the same for the labels
    tags = np.array([l.split() for l in labels])  
    tag_output = [
        [tag2id[sen[t]] if t < seq_lengths[idx] else pad_id for t in range(max_len)]
            for idx, sen in enumerate(tags)]
    # Replace words of the input with <unk> with p = word_dropout.
    if word_dropout > 0.:
        unk_id = vocab[UNK_TOKEN]
        word_drop =  [
            [unk_id if (np.random.random() < word_dropout and t < seq_lengths[idx]) else word_ids[t] for t in range(max_len)] 
                for idx, word_ids in enumerate(pad_id_input)]
    
    # The output batch is shifted by 1.
    pad_id_output = [
        [vocab[sen[t+1]] if t < seq_lengths[idx] else pad_id for t in range(max_len)]
            for idx, sen in enumerate(tok)]
    
    # Convert everything to PyTorch tensors
    batch_input = torch.tensor(pad_id_input) 
    batch_output = torch.tensor(pad_id_output)
    tags = torch.tensor(tag_output)
    # define sequence mask to know what is a word and what is padding
    # this is used to mask the loss and we do not end up taking into account empty sequences
    seq_mask = (batch_input != vocab[PAD_TOKEN])
    seq_length = torch.tensor(seq_lengths)
    
    # Move all tensors to the given device 
    batch_input = batch_input.to(device) 
    batch_output = batch_output.to(device) 
    seq_mask = seq_mask.to(device) 
    seq_length = seq_length.to(device) 
    tags = tags.to(device)
    
    return batch_input, batch_output, tags, seq_mask, seq_length 

def get_batch(source, i, bptt, output_target=True):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    if output_target:           
      return data, target
    else:
      return data

def get_batch_tup(source, i, bptt):
    x, y = get_batch(source[0], i, bptt) # Get a batch of the sentences
    tags = get_batch(source[1], i, bptt, output_target=False) # Get a batch of the labels
    return ((x, y), tags)

def get_loaders_tok(source, bs, bptt, vocab, tag2id, device, word_dropout=0., use_var_bptt=False): 
    # sentences, labels = source 
    #data = batchify(source, bs)
    data = batchify_tup(source, bs)
    loader = []    
    i = 0
    while i < data[0].size(0) - 2:
        if use_var_bptt:
            rand_bptt = bptt if np.random.random() < 0.95 else bptt / 2. 
            seq_len = max(5, int(np.random.normal(rand_bptt, 5))) 
        else:
            seq_len = bptt
        batch = get_batch_tup(data, i, seq_len)
        loader.append(batch) # batch = ((x,y), tags)       
        i += seq_len

        # # sentences, labels = data
        # x, y, tags, seq_mask, seq_length = create_batch(data, vocab, tag2id, device, word_dropout=0.)
        # # convert labels into tensor
        # # labels = data[1]
        # # labels = np.array([l.split() for l in labels]) 
        # # labels = torch.tensor(labels)       
        # # divide into batches 
        # x_batch = batchify(x, bs)
        # y_batch = batchify(y, bs)
        # #tags    = batchify(tags, bs)   # Tags are currently set up for if we want to train with them (we dont tho)
        # #labels  = batchify(labels, bs) 
        # seq_mask_batch = batchify(seq_mask, bs)        
        # batch = (x_batch, y_batch)
        # loader.append((batch, tags))   # batch = x, y, currently not using seq_mask, seq_length
    
    return loader

def get_loaders(source, bs, bptt, use_var_bptt=False):
    data = batchify(source, bs)
    loader = []
    
    i = 0
    while i < data.size(0) - 2:
        if use_var_bptt:
            rand_bptt = bptt if np.random.random() < 0.95 else bptt / 2.
            seq_len = max(5, int(np.random.normal(rand_bptt, 5)))
        else:
            seq_len = bptt
        
        batch = get_batch(data, i, seq_len, output_target=True)
        loader.append(batch)
        i += seq_len
    
    return loader

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    
def drop_mult(model, dm):
    for i in range(len(model.encoder.rnn)):
        model.encoder.weight_dp[i].weight_p *= dm
    model.encoder.emb_dp.embed_p *= dm
    model.encoder.hidden_dp.p *= dm
    model.encoder.input_dp.p *= dm
    return model

def get_param_groups(model):
    p_groups = [{'name': '0', 'params': []}, {'name': '1', 'params': []}]
    for n, p in model.named_parameters():
        if 'rnn' in n:
            p_groups[1]['params'].append(p)
        else:
            p_groups[0]['params'].append(p)
    return p_groups

def extract_tags(p, id2tag): 
    # Extracts tags into a list
    if len(p) == 1:
        tag = [id2tag[p.item()]]
    else:           
        tag = [id2tag[q.item()] for q in p] 
    return tag

def accuracy(out, y):
    return torch.sum(torch.max(torch.softmax(out, dim=1), dim=1)[1] == y).item() / len(y)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def vectorize(text, word2idx, vocab_set, msl):
    v_text = [word2idx[word] if word in vocab_set else word2idx['<unk>'] for word in text]
  
    v_text = v_text[:msl]

    if len(v_text) < msl:
        v_text = v_text + [word2idx['<pad>'] for _ in range(msl - len(v_text))]
  
    return v_text

def produce_dataloaders(X_train, y_train, X_val, y_val, word2idx, vocab_set, msl, bs, drop_last=True):
    X_train =  [vectorize(text, word2idx, vocab_set, msl) for text in tqdm(X_train)]
    X_val =  [vectorize(text, word2idx, vocab_set, msl) for text in tqdm(X_val)]
    
    X_train = torch.LongTensor(X_train)
    X_val = torch.LongTensor(X_val)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    
    train_set = data_utils.TensorDataset(X_train, y_train)
    val_set = data_utils.TensorDataset(X_val, y_val)

    train_loader = data_utils.DataLoader(train_set, bs, drop_last=drop_last)
    val_loader = data_utils.DataLoader(val_set, bs, drop_last=drop_last)
    
    return train_loader, val_loader
