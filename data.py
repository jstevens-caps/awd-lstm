import os   
from io import open   
import torch   
from torch.utils.data import Dataset
import codecs
import re

from utils import Data_and_y_handler

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
tag_ids = {'MISC':0, 'LOC':1, 'PER':2, 'ORG':3, 'O':4}

class TextDataset(Dataset): #You make sentences  
    """ 
       Loads a list of sentences into memory from a text file, 
       split by newlines. 
    """ 
    def __init__(self, path, input_file, vocabulary, max_len=50, skip_dict=False, evalu=False):  
        # by default the max number of words in a sentence is 50 words  
        # we avoid processing very large mini-batches  
        self.vocabulary = vocabulary
        #print("Loading file: ", input_file)
        if not skip_dict:
            self.vocabulary = from_data(os.path.join(path, input_file), self.vocabulary)   
        self.data = []                  # sentences (ids) tensor  
        self.y = []                     # labels tensor
        self.sentence_words_total = []  # sentences (words) tensor

        # open uft-8 files  
        with codecs.open(os.path.join(path, input_file), 'r', encoding='utf8')  as f:  
            sentence = [self.vocabulary.word2idx[SOS_TOKEN]]  
            labels = ['O']  # test_true, random number for class
            sentence_words = [SOS_TOKEN]
            for line in f:  
                line = line.strip()  
                    
                if line: 
                  #(token, tag) = re.split("\t", line)
                  (word, word_type, tag) = re.split(" ", line)
                  sentence.append(self.vocabulary.word2idx[word if word in self.vocabulary.vocab_set else UNK_TOKEN])
                  sentence_words.append(word)
                  labels.append(tag) 
                else: 
                  if len(sentence) > 0 and len(sentence) <= max_len: 
                        sentence.append(self.vocabulary.word2idx[EOS_TOKEN]) 
                        sentence_words.append(EOS_TOKEN)                  
                        labels.append('O') # number to fill for the EOS token
                        #print("sentence", sentence_words)
                        # data.append(torch.tensor(sentence).type(torch.int64)) 
                        # y.append(torch.tensor(labels).type(torch.int64))
                        #self.data.append(sentence_words)
                        self.y.append(' '.join(labels)) 
                        self.data.append(' '.join(sentence_words))
                  sentence = [self.vocabulary.word2idx[SOS_TOKEN]] 
                  labels = ['O'] # Number associated with random class 
                  sentence_words = [SOS_TOKEN]                    
            #data_ten = torch.cat(data)
            #y_ten = torch.cat(y)
            #print("data_ten", data_ten[:100])          
        # if evalu:
        #   return data_ten, y_ten, sentence_words_total
        # return data_ten, y_ten
                    

    def __len__(self):
        # overide len to get number of instances
        return len(self.data)

    def __getitem__(self, idx):
        # get words and label for a given instance index
        return self.data[idx], self.y[idx]

class Vocabulary(object):
    # may need to add something here in lines of dealing with tags
    def __init__(self):
        self.word2idx = {'<unk>': 0, '<pad>': 1, '<eos>': 2, '<sos>': 3}
        self.idx2word = ['<unk>', '<pad>', '<eos>', '<sos>']
        self.vocab_set = set(self.idx2word)

    def __getitem__(self, key):
      # map words to indices, used for the embedding look-up table
        return self.word2idx[key] if key in self.word2idx else self.word2idx[UNK_TOKEN]

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.vocab_set.add(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
    
    def from_data(input_files, _):  # This function is currently not being calle
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
                      if token not in vocab.word2idx:
                        idx = len(vocab.word2idx)
                        vocab.word2idx[token] = idx
                        vocab.idx2word.append(token) 
        return vocab

def from_data(input_file, vocab): 
        # Load unique words from data 
        # vocab = Vocabulary() 
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
                    # print(idx)
                    #print("adding to vocabulary token", token)
                    vocab.word2idx[token] = idx
                    vocab.idx2word.append(token)
                    vocab.vocab_set.add(token)
                    # print("index of word in word2idx", vocab.word2idx[token])
                    # print("word at idx in idx2word", vocab.idx2word[idx])
        return vocab

class Corpus_tok(object): # Corpus object for CONLL dataset with labels
    def __init__(self, path, train_file, valid_file, test_file, load_vocab=False, vocab_file='vocab.pth', max_sen_len=50):
        self.vocabulary = Vocabulary()
        if load_vocab: 
            with open(os.path.join(path, vocab_file), 'rb') as f:
                word2idx, idx2word = torch.load(f)
            self.vocabulary.word2idx = word2idx
            self.vocabulary.idx2word = idx2word
            self.vocabulary.vocab_set = set(idx2word)

        # This self.train/valid/test now consist of (sentences, labels), sentences split by newlines 
        train_tmp = self.make_los(os.path.join(path, train_file), skip_dict=load_vocab, max_len=max_sen_len) 
        valid_tmp = self.make_los(os.path.join(path, valid_file), skip_dict=load_vocab,  max_len=max_sen_len) 
        test_tmp = self.make_los(os.path.join(path, test_file), skip_dict=load_vocab, max_len=max_sen_len)
        self.train = Data_and_y_handler(train_tmp[0], train_tmp[1])
        self.valid = Data_and_y_handler(valid_tmp[0], valid_tmp[1])
        self.test = Data_and_y_handler(test_tmp[0], test_tmp[1])
        # self.test = sen_ids, labs
        # self.test_sentences = sen_words
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
                    self.vocabulary.add_word(word)

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
                    ids.append(self.vocabulary.word2idx[word if word in self.vocabulary.vocab_set else UNK_TOKEN])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


#        Loads a list of sentences into memory from a text file, 
#        split by newlines. 
    
    def make_los(self, input_file, max_len=50, skip_dict=False, evalu=False):  
        # by default the max number of words in a sentence is 50 words  
        # we avoid processing very large mini-batches  
        print("Loading file: ", input_file)
        if not skip_dict:
            self.vocabulary = from_data(input_file, self.vocabulary)   
        data = []                  # sentences (ids) tensor  
        y = []                     # labels tensor
        sentence_words_total = []  # sentences (words) tensor

        # open uft-8 files  
        with codecs.open(input_file, 'r', encoding='utf8')  as f:  
            sentence = [self.vocabulary.word2idx[SOS_TOKEN]]  
            labels = [4]  # test_true, random number for class
            sentence_words = [SOS_TOKEN]
            for line in f:  
                line = line.strip()  
                    
                if line: 
                  #(token, tag) = re.split("\t", line)
                  (word, word_type, tag) = re.split(" ", line)
                  sentence_words.append(word)
                  sentence.append(self.vocabulary.word2idx[word if word in self.vocabulary.vocab_set else UNK_TOKEN])
                  labels.append(tag_ids[tag]) 
                else: 
                  if len(sentence) > 0 and len(sentence) <= max_len: 
                        sentence.append(self.vocabulary.word2idx[EOS_TOKEN]) 
                        sentence_words.append(EOS_TOKEN)
                        sentence_words_total.append(sentence_words)
                        labels.append(4) # number to fill for the EOS token
                        #print("sentence", sentence_words)
                        # data.append(torch.tensor(sentence).type(torch.int64)) 
                        # y.append(torch.tensor(labels).type(torch.int64))
                        data.append(sentence)
                        y.append(sentence) 
                  sentence = [self.vocabulary.word2idx[SOS_TOKEN]] 
                  labels = [4] # Number associated with random class 
                  sentence_words = [SOS_TOKEN]                    
            #data_ten = torch.cat(data)
            #y_ten = torch.cat(y)
            #print("data_ten", data_ten[:100])          
        # if evalu:
        #   return data_ten, y_ten, sentence_words_total
        # return data_ten, y_ten
        #print("len data", len(data))
        if evalu:
          return data, y, sentence_words_total
        return data, y  

    def __len__(self):
        # overide len to get number of instances
        return len(self.data)

    def __getitem__(self, idx):
        # get words and label for a given instance index
        return self.data[idx], self.y[idx]
    
class Corpus(object): # Corpus object for not tokenized text files
    def __init__(self, path, train_file, valid_file, test_file, load_vocab=False, vocab_file='vocab.pth'):
        self.vocabulary = Vocabulary()
        if load_vocab:
            with open(os.path.join(path, vocab_file), 'rb') as f:
                word2idx, idx2word = torch.load(f)
            self.vocabulary.word2idx = word2idx
            self.vocabulary.idx2word = idx2word
            self.vocabulary.vocab_set = set(idx2word)
        self.train = self.tokenize(os.path.join(path, train_file), skip_dict=load_vocab)
        self.valid = self.tokenize(os.path.join(path, valid_file), skip_dict=load_vocab)
        self.test = self.tokenize(os.path.join(path, test_file), skip_dict=load_vocab)
        
    def build_dict(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            words = []
            for line in f:
                if line:
                  (word, word_type, tag) = re.split(" ", line)
                  words.append(word)
                else:
                  for word in words:
                      self.vocabulary.add_word(word)
                  words = []

    def tokenize(self, path, skip_dict=False):
        if not skip_dict:
            self.build_dict(path)
        
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            words = []
            for line in f:
                if line: 
                  (word, word_type, tag) = re.split(" ", line)
                  words.append(word)
                else:
                  if len(sentence) > 0 and len(sentence) <= 50: 
                    ids = []
                    words.append([EOS_TOKEN])
                    for word in words:
                      ids.append(self.vocabulary.word2idx[word if word in self.vocabulary.vocab_set else '<unk>'])
                    idss.append(torch.tensor(ids).type(torch.int64))
                  words = []
            ids = torch.cat(idss)

        return ids
