import os

import random
import logging
from collections import Counter
import codecs
import glob
import json


import numpy as np

import pickle

logger = logging.getLogger(__name__)

class Vocabulary(object):
    def __init__(self, vocab_file, dim_emb=0):
        with open(vocab_file, 'rb') as f:
            self.size, self._word2id, self._id2word = pickle.load(f)

    def word2id(self, word):
        if word not in self._word2id:
          return self._word2id['<unk>']
        return self._word2id[word]

    def id2word(self, word_id):
        if word_id >= self.size:
          raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id2word[word_id]

def build_vocab(data_path, save_path, min_occur=1):
    word2id = {'<pad>':0, '<go>':1, '<eos>':2, '<unk>':3}
    id2word = ['<pad>', '<go>', '<eos>', '<unk>']
    words = []




    with open(data_path, 'r') as f:
        while True:
            string_ = f.readline()
            if not string_: break
            dict_example = json.loads(string_)
            sent = dict_example["review"]
            words += sent.split()

    cnt = Counter(words)
    for word in cnt:
        if cnt[word] >= min_occur:
            word2id[word] = len(word2id)
            id2word.append(word)
    vocab_size = len(word2id)
    with open(save_path, 'wb') as f:
        pickle.dump((vocab_size, word2id, id2word), f, pickle.HIGHEST_PROTOCOL)
