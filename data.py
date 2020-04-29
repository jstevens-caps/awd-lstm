import os   
from io import open   
import torch   
from torch.utils.data import Dataset

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

class Vocabulary: #this is currently not being called 
    """
        Creates a vocabulary from a list of data files. The data files have been
        tokenized and pre-processed.
    """
    def __init__(self):
        # mapping from tokens/words into indices and otherwise
        self.idx_to_word = {0: PAD_TOKEN, 1: UNK_TOKEN, 2: SOS_TOKEN, 3: EOS_TOKEN}
        self.word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1, SOS_TOKEN: 2, EOS_TOKEN: 3}
        # we can also get the frequency of words, a common practice is to ignore low frequency words
        self.word_freqs = {}
    
    def __getitem__(self, key):
        # map words to indices, used for the embedding look-up table
        return self.word_to_idx[key] if key in self.word_to_idx else self.word_to_idx[UNK_TOKEN]
    
    def word(self, idx): 
        # map an index to word 
        return self.idx_to_word[idx] 
    
    def size(self): 
        # number of unique words 
        return len(self.word_to_idx) 
    
    
    def from_data(input_files): 
        # Load unique words from data
        vocab = Vocabulary() 
        for input_file in input_files: 
            with codecs.open(input_file, 'r', encoding='utf8')  as f: 
                for line in f: 
                    
                    # Strip whitespace and the newline symbol.
                    line = line.strip()
                    # split data from .tsv  
                    if line:
                      #(token, tag) = re.split("\t", line)
                      (token, word_type, tag) = re.split(" ", line)
                      if token not in vocab.word_to_idx:
                        idx = len(vocab.word_to_idx)
                        vocab.word_to_idx[token] = idx
                        vocab.idx_to_word[idx] = token
                        
                            
        return vocab

class TextDataset(Dataset): #This is currently not being called 
    """ 
       Loads a list of sentences into memory from a text file, 
       split by newlines. 
    """ 
    def __init__(self, input_files, max_len=50):  
        # by default the max number of words in a sentence is 50 words  
        # we avoid processing very large mini-batches  
        self.data = [] 
        self.y  = [] 
        for input_file in input_files:  
            # open uft-8 files  
            with codecs.open(input_file, 'r', encoding='utf8')  as f:  
                sentence = []  
                labels = []  
                start_flag = 0  
                for line in f:  
                    line = line.strip()  
                    
                    if line: 
                      #(token, tag) = re.split("\t", line)
                      (token, word_type, tag) = re.split(" ", line)
                      sentence.append(token)
                      labels.append(tag[0])   #altered this
                    else: 
                      if len(sentence) > 0 and len(sentence) <= max_len:
                            self.data.append(' '.join(sentence))
                            self.y.append(' '.join(labels))
                      sentence = []
                      labels = []
         

    def __len__(self):
        # overide len to get number of instances
        return len(self.data)

    def __getitem__(self, idx):
        # get words and label for a given instance index
        return self.data[idx], self.y[idx]

class Dictionary(object):
    # may need to add something here in lines of dealing with tags
    def __init__(self):
        self.word2idx = {'<unk>': 0, '<pad>': 1, '<eos>': 2}
        self.idx2word = ['<unk>', '<pad>', '<eos>']
        self.vocab_set = set(self.idx2word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.vocab_set.add(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
    
    def from_data(input_files): 
        # Load unique words from data
        vocab = Dictionary() 
        for input_file in input_files: 
            with codecs.open(input_file, 'r', encoding='utf8')  as f: 
                for line in f: 
                    
                    # Strip whitespace and the newline symbol.
                    line = line.strip()
                    # split data from .tsv  
                    if line:
                      #(token, tag) = re.split("\t", line)
                      (token, word_type, tag) = re.split(" ", line)
                      if token not in vocab.word2idx:
                        idx = len(vocab.word2idx)
                        vocab.word2idx[token] = idx
                        vocab.idx2word[idx] = token
        return vocab
                        
class Corpus_tok(object):
    def __init__(self, path, train_file, valid_file, test_file, load_vocab=False, vocab_file='vocab.pth', max_sen_len=50):
        self.dictionary = Dictionary()
        if load_vocab:
            with open(os.path.join(path, vocab_file), 'rb') as f:
                word2idx, idx2word = torch.load(f)
            self.dictionary.word2idx = word2idx
            self.dictionary.idx2word = idx2word
            self.dictionary.vocab_set = set(idx2word)

        # This self.train/valid/test now consist of (sentences, labels), sentences split by newlines 
        self.train = make_los(os.path.join(path, train_file), skip_dict=load_vocab, max_len=max_sen_len) 
        self.valid = make_los(os.path.join(path, valid_file), skip_dict=load_vocab,  max_len=max_sen_len) 
        self.test  = make_los(os.path.join(path, test_file), skip_dict=load_vocab, max_len=max_sen_len)
#         self.train = self.tokenize(os.path.join(path, train_file), skip_dict=load_vocab)
#         self.valid = self.tokenize(os.path.join(path, valid_file), skip_dict=load_vocab)
#         self.test = self.tokenize(os.path.join(path, test_file), skip_dict=load_vocab)
        
    def build_dict(self, path):  # no longer being called
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

    def tokenize(self, path, skip_dict=False):   # no longer being called
        if not skip_dict:
            self.build_dict(path)
        
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word if word in self.dictionary.vocab_set else '<unk>'])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
    
   
#        Loads a list of sentences into memory from a text file, 
#        split by newlines. 
    
    def make_los(self, input_files, max_len=50, skip_dict=False):  
        # by default the max number of words in a sentence is 50 words  
        # we avoid processing very large mini-batches  
        if not skip_dict:
            self.dictionary.from_data(input_files)
        data = [] # sentences
        y = []    # labels
        for input_file in input_files:  
            # open uft-8 files  
            with codecs.open(input_file, 'r', encoding='utf8')  as f:  
                sentence = []  
                labels = []  
                start_flag = 0  
                for line in f:  
                    line = line.strip()  
                    
                    if line: 
                      #(token, tag) = re.split("\t", line)
                      (token, word_type, tag) = re.split(" ", line)
                      sentence.append(token)
                      labels.append(tag)   #altered this
                    else: 
                      if len(sentence) > 0 and len(sentence) <= max_len:
                            self.data.append(' '.join(sentence))
                            self.y.append(' '.join(labels))
                      sentence = []
                      labels = []
         
        return data, y

    def __len__(self):
        # overide len to get number of instances
        return len(self.data)

    def __getitem__(self, idx):
        # get words and label for a given instance index
        return self.data[idx], self.y[idx]
    
class Corpus(object):
    def __init__(self, path, train_file, valid_file, test_file, load_vocab=False, vocab_file='vocab.pth'):
        self.dictionary = Dictionary()
        if load_vocab:
            with open(os.path.join(path, vocab_file), 'rb') as f:
                word2idx, idx2word = torch.load(f)
            self.dictionary.word2idx = word2idx
            self.dictionary.idx2word = idx2word
            self.dictionary.vocab_set = set(idx2word)
        self.train = self.tokenize(os.path.join(path, train_file), skip_dict=load_vocab)
        self.valid = self.tokenize(os.path.join(path, valid_file), skip_dict=load_vocab)
        self.test = self.tokenize(os.path.join(path, test_file), skip_dict=load_vocab)
        
    def build_dict(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

    def tokenize(self, path, skip_dict=False):
        if not skip_dict:
            self.build_dict(path)
        
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word if word in self.dictionary.vocab_set else '<unk>'])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
