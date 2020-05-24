# AWD-LSTM and AVITM

This repository contains the code for the capstone project of Jasper Stevens, supervised by Dr. Miguel Rios Goana. 
The code in this repository allows you to train a latent variable model used to predict named entity types on the CONLL Dutch dataset.
This code is inspired by the following papers:
* [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182) (Merity et al., 2017).
* [Autoencoding Variational Inference for Topic Models](https://arxiv.org/pdf/1703.01488) (Srivastava & Sutton, 2017) 

# Requirements
Libraries you need:
* PyTorch - At least v.1.0.0
* [Transformers](https://github.com/huggingface/transformers) - The learning rate scheduler we use comes from there
* spacy - We use spacy tokenizers for ULMFiT
* numpy and pandas - For random numbers and data manipulation
* tqdm - Progress bars to keep our sanity intact :)

Hardware you need:
* A GPU with at least 11 GB - All the results in this repository have been produced on machines with NVIDIA GTX 1080Ti and servers with NVIDIA Tesla P100 GPUs. A lot of the models when training do take about 10 GB of memory. You *could* get away with a smaller and slower GPU but do note that this affects your speed and performance.

# Using the Training Scripts
Here are a few examples 

``` 
python awd-lstm/main.py --rebuild_dataset --lr=1e-3 --num-topic=5 --epochs=2 --bs=80 --bptt=80 --eval_bs=80 --num_layers=3 --optimizer='adam' --en1-units=400 --en2-units=400 --path=awd-lstm/data/conll2002 --train=ned.train_mod --valid=ned.testb_mod --test=ned.testa_mod --output=pretrained_wt2_jasper --tie_weights --save_vocab --vocab_file=vocab.pth --gpu=0
```

# Generation
You can use your pretrained models to generate text. Here is an example:
```
python awd-lstm/generate.py --path data/wikitext-2 --tie_weights --vocab_file=vocab.pth --pretrained_file=pretrained_wt103.pth --temp=0.7 --nwords=100
```

# Finetuning Language Models
To finetune on a dataset, you'll need the saved vocabulary file and the pretrained weights. For text datasets, you will need to preprocess them such that each sample is separated by a blank line (the code replaces this with an ```<eos>``` token) and each sample has been pre-tokenized (the code should only need to do ```.split()``` to produce tokens). 

Here's an example finetuning the iMDB dataset on a pretrained model trained using WikiText-103. We first freeze the model and finetune only the last layer:

```
python awd-lstm/main.py --path=data --train=imdb/lm_data/train.txt --valid=imdb/lm_data/valid.txt --test=imdb/lm_data/test.txt --output=imdb/lm_data/imdb_finetuned_part --bs=70 --bptt=70 --epochs=10 --tie_weights --load_vocab --vocab_file=pretrained_wt103/vocab.pth --use_pretrained --pretrained_file=pretrained_wt103/pretrained_wt103.pth --freeze_encoder --optimizer=adam --no_lr_scaling --lr=1e-2 --gpu=0
```

Then we take that and finetune it for 10 full epochs unfrozen.

```
python awd-lstm/main.py --path=data --train=imdb/lm_data/train.txt --valid=imdb/lm_data/valid.txt --test=imdb/lm_data/test.txt --output=imdb/lm_data/imdb_finetuned_full --bs=70 --bptt=70 --epochs=10 --tie_weights --load_vocab --vocab_file=pretrained_wt103/vocab.pth --use_pretrained --pretrained_file=imdb/lm_data/imdb_finetuned_part.pth --optimizer=adam --lr=1e-3 --gpu=0
```

If you need an example for how to preprocess data, I provide a version of the iMDB Sentiments dataset [here](https://www.kaggle.com/jcblaise/imdb-sentiments). The .csv files are for classification and the .txt files are for language model finetuning.

I cannot, at the moment, provide my own pretrained WikiText-103 models. For the results involving pretrained models, I adapted the pretrained weights provided by [FastAI](https://www.fast.ai/) for now (compatible checkpoint [here](https://storage.googleapis.com/blaisecruz/ulmfit/pretrained_wt103.zip)). More details are in the **To-do** section below.

# ULMFiT / Finetuning Classifiers
To finetune a classifier, make sure you have a finetuned language model at hand. Load the ```ULMFiT.ipynb``` notebook and follow the instructions there.

# Using Layers
The ```layers.py``` file provides some layers you can use outside of this project.

Included so far are:
* Two encoder layers - ```AWDLSTMEncoder``` and ```LSTMEncoder```
* Two language modeling decoder layers - ```DropoutLinearDecoder``` and ```LinearDecoder```
* One classification decoder layer - ```ConcatPoolingDecoder```

For language modeling, a ```RNNModel``` wrapper is provided. Pass in an encoder and a decoder and you're good to go. For example:

```python
encoder = AWDLSTMEncoder(vocab_sz=vocab_sz, emb_dim=400, hidden_dim=1152, num_layers=3, tie_weights=True)
decoder = DropoutLinearDecoder(hidden_dim=400, vocab_sz=vocab_sz) # Using 400 since we're tying weights
model = RNNModel(encoder, decoder, tie_weights=True).to(device)

# Your code here
```

The encoders in the API are written as standalone and independent - they can ouput any number of parameters. The decoders in the API are written to be able to handle any number of parameters given to them. This allows the ```RNNModel``` to act like a wrapper - you can mix and match encoders and decoders. Please refer to ```layers.py``` for more information

Classification is very similar, using the ```RNNClassifier``` wrapper. For example:

```python
encoder = AWDLSTMEncoder(vocab_sz=vocab_sz, emb_dim=400, hidden_dim=1152, num_layers=3)
decoder = ConcatPoolingDecoder(hidden_dim=400, bneck_dim=50, out_dim=2)
model = RNNClassifier(encoder, decoder).to(device)

# Your code here
```

The ```RNNClassifier``` wrapper also has class functions for freezing and unfreezing layers (for ULMFiT). For now, it's only tested to work without errors with the ```AWDLSTMEncoder```, since it was designed for it.

The encoders and decoders can also be used without the ```RNNModel``` and ```RNNClassifier``` wrappers, should you want to. You can use them inside your own models, like the ```AWDLSTMEncoder``` for example:

```python

class MyModel(nn.Module):
    def __init__(self, ...):
        super(MyModel, self).__init__()
        self.encoder = AWDLSTMEncoder(...)

        # Your code here

```

# Changelog
Version 0.3
* Added ULMFiT and related code
* Added finetuning and pretraining capabilities
* Added a way to load vocabularies
* Fixed the generation script

Version 0.2
* Added basic LSTM Encoder support and modularity
* Added an option to train with Adam with Linear Warmups
* Fixed parameters and reduced required parameters
* AR/TAR only activates when using an AWD-LSTM

Version 0.1
* Added basic training functionality
