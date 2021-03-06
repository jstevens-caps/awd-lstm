import torch
import torch.utils.data as data_utils
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader

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

def create_batch(sentences, labels, vocab, tag2id, device, word_dropout=0.):
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
    tok = np.array([(sen.split()) for sen in sentences])
    seq_lengths = [len(sen)-1 for sen in tok] 
    max_len = max(seq_lengths) 
    pad_id = vocab[PAD_TOKEN]  #pad token 
     # padding of the sentences that a sorther than the max with 0
    pad_id_input = [
        [vocab[sen[t]] if t < seq_lengths[idx] else pad_id for t in range(max_len)]
            for idx, sen in enumerate(tok)]
    # we do the same for the labels
    tags = np.array([l.split() for l in labels])  
    #tags = labels
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
    batch_input = torch.tensor(pad_id_input).transpose(0,1)
    batch_output = torch.tensor(pad_id_output).transpose(0,1).reshape((-1,))
    tags = torch.tensor(tag_output)
    # define sequence mask to know what is a word and what is padding
    # this is used to mask the loss and we do not end up taking into account empty sequences
    #seq_mask = (batch_input != vocab[PAD_TOKEN])
    seq_length = torch.tensor(seq_lengths)
    
    # Move all tensors to the given device 
    batch_input = batch_input.to(device) 
    batch_output = batch_output.to(device) 
    #seq_mask = seq_mask.to(device) 
    seq_length = seq_length.to(device) 
    tags = tags.to(device)
    
    return batch_input, batch_output, tags #, seq_mask, seq_length 

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
    return source
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

def get_word_list(tup_list, to_search, rebuild=False):
  if rebuild:
    list_o_tup = []
    for i in tup_list:
      for j in range(len(i[0])):
        list_o_tup.append((i[0][j],i[1][j]))
    topic_words = [word for word, tag in list_o_tup if (tag == to_search)]
  else:
    topic_words = [word for word, tag in tup_list if (tag == to_search)]
    list_o_tup = []
  return topic_words, list_o_tup

def _get_coherence(best_components, n_components, idx2token, k=10, topics=5):
        """Get coherence using palmetto web service."""
        component_dists = best_components
        base_url = 'http://palmetto.aksw.org/palmetto-webapp/service/cv?words='
        scores = []
        i = 0
        while i < topics:
            t = np.random.randint(0, n_components)
            _, idxs = torch.topk(component_dists[t], k)
            component_words = [idx2token[idx]
                               for idx in idxs.cpu().numpy()]
            url = base_url + '%20'.join(component_words)
            try:
                score = float(requests.get(url, timeout=300).content)
                scores += [score]
                i += 1
            except requests.exceptions.Timeout:
                print("Attempted scoring timed out.  Trying again.")
                continue
        return np.mean(scores)

def get_topics(best_components, n_components, idx2token, k=10):
        """
        Retrieve topic words.
        Args
            k : (int) number of words to return per topic, default 10.
        """
        assert k <= self.input_size, "k must be <= input size."
        component_dists = best_components
        topics = defaultdict(list)
        for i in range(n_components):
            _, idxs = torch.topk(component_dists[i], k)
            component_words = [idx2token[idx]
                               for idx in idxs.cpu().numpy()]
            topics[i] = component_words
        return topics
    
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

class Data_and_y_handler(Dataset):
    def __init__(self, data, y):
        self.data = data
        self.y = y 
    def __len__(self):
        return len(data)
    def __getitem__(self, idx):
        # get words and label for a given instance index
        return self.data[idx], self.y[idx]

class SortingTextDataLoader:
    """
    A wrapper for the DataLoader class that sorts sentences by their
    lengths in descending order.
    """

    def __init__(self, dataloader):
        # we sort sentences for the optimization of the RNN code!
        self.dataloader = dataloader
        self.it = iter(dataloader)
    
    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dataloader)
    
    def __next__(self):
        sentences = None
        labels = None
        for s,l in self.it:
            sentences = s
            labels = l
            break

        if sentences is None:
            self.it = iter(self.dataloader)
            raise StopIteration
        if labels is None:
            self.it = iter(self.dataloader)
            raise StopIteration
        
        sentences = np.array(sentences)
        labels = np.array(labels)
        sort_keys = sorted(range(len(sentences)), 
                           key=lambda idx: len(sentences[idx].split()), 
                           reverse=True)
        sorted_sentences = sentences[sort_keys]
        sorted_labels = labels[sort_keys]
        return sorted_sentences, sorted_labels

    # def __next__(self):
    #     sentences = None
    #     labels = None
    #     for s,l in self.it:
    #         sentences = s
    #         labels = l
    #         break

    #     if sentences is None:
    #         self.it = iter(self.dataloader)
    #         raise StopIteration
    #     if labels is None:
    #         self.it = iter(self.dataloader)
    #         raise StopIteration
        
    #     # Weird error, seems like sentences and labels are identical, 0 labels 1 sentences
    #     #print("sentences", sentences)
    #     #print("labels", labels)
    #     sentences = np.array(sentences[1])
    #     labels = np.array(labels[0])
    #     sort_keys = sorted(range(len(sentences)), 
    #                        #key=lambda idx: len(sentences[idx].split()), 
    #                        reverse=True)
    #     sorted_sentences = sentences[sort_keys]
    #     sorted_labels = labels[sort_keys]
    #     return (sorted_sentences, sorted_labels)
