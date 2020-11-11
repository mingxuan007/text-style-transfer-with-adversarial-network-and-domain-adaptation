import os
import random
import logging
import sys
import tensorflow as tf
import numpy as np



logger = logging.getLogger(__name__)
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from scipy.spatial.distance import cosine
import statistics






class Accumulator(object):
    def __init__(self, div, names):
        self.div = div
        self.names = names
        self.n = len(self.names)
        self.values = [0] * self.n

    def clear(self):
        self.values = [0] * self.n

    def add(self, values):
        for i in range(self.n):
            self.values[i] += values[i] / self.div

    def output(self, prefix,write_dict, mode):
        if prefix:
            prefix += ' '
        for i in range(self.n):
            prefix += '%s %.2f' % (self.names[i], self.values[i])
            if i < self.n - 1:
                prefix += ', '
        logger.info(prefix)

        add_summary_value(write_dict['writer'], self.names, self.values, write_dict['step'], mode)


def add_summary_value(writer,keys, values, iteration, mode, domain=''):
    if mode not in ['train', 'valid']:
        return

    for key, value in zip(keys, values):
        key = os.path.join(mode, domain, key)
        summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
        writer.add_summary(summary, iteration)
    writer.flush()
    
def strip_eos(sents):
    new_ids, lengths = [], []
    for sent in sents:
        if '<eos>' in sent:
            sent = sent[:sent.index('<eos>')]
        new_ids.append(sent)
        lengths.append(len(sent))
    return new_ids, lengths



def write_output_v0(origin, transfer, reconstruction, path,epoch):
    t = open(path + str(epoch)+ '_transfer.txt', 'w')
    r = open(path + str(epoch)+ '_reconstruction.txt', 'w')
    for i in range(len(origin)):
        try:
            output = origin[i] + '\t' + transfer[i] + '\n'
            t.write(output)
        except:
            pass
        try:
            output = origin[i] + '\t' + reconstruction[i] + '\n'
            r.write(output)
        except:
            pass
    t.close()
    r.close()





def load_glove_model(glove_file):
    logger.debug("Loading Glove Model")
    model = dict()
    with open(glove_file,encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
        logger.debug("Done. {} words loaded!".format(len(model)))
    return model
glove_path='./glove.6B.100d.txt'    
model=load_glove_model('glove_path')

def get_sentence_embedding(tokens, model):
    embeddings = np.asarray([model[token] for token in tokens if token in model])

    min_embedding = np.min(embeddings, axis=0)
    max_embedding = np.max(embeddings, axis=0)
    mean_embedding = np.mean(embeddings, axis=0)
    sentence_embedding = np.concatenate([min_embedding, max_embedding, mean_embedding], axis=0)

    return sentence_embedding


def get_content_preservation_score(actual_word_lists, generated_word_lists, embedding_model):
    sentiment_words = get_style_words()
    cosine_distances = list()
    skip_count = 0
    for word_list_1, word_list_2 in zip(actual_word_lists, generated_word_lists):
        cosine_similarity = 0
        words_1 = set(word_list_1)
        words_2 = set(word_list_2)

        words_1 -= sentiment_words
        words_2 -= sentiment_words
        try:
            cosine_similarity = 1 - cosine(
                get_sentence_embedding(words_1, embedding_model),
                get_sentence_embedding(words_2, embedding_model))
            cosine_distances.append(cosine_similarity)
        except ValueError:
            skip_count += 1
            logger.debug("Skipped lines: {} :-: {}".format(word_list_1, word_list_2))

    logger.debug("{} lines skipped due to errors".format(skip_count))
    mean_cosine_distance = statistics.mean(cosine_distances) if cosine_distances else 0

    del sentiment_words

    return mean_cosine_distance


def get_word_overlap_score(actual_word_lists, generated_word_lists):
    english_stopwords = get_stopwords()
    sentiment_words = get_style_words()

    scores = list()
    for word_list_1, word_list_2 in zip(actual_word_lists, generated_word_lists):
        score = 0
        words_1 = set(word_list_1)
        words_2 = set(word_list_2)

        words_1 -= sentiment_words
        words_1 -= english_stopwords
        words_2 -= sentiment_words
        words_2 -= english_stopwords

        word_intersection = words_1 & words_2
        word_union = words_1 | words_2
        if word_union:
            score = len(word_intersection) / len(word_union)
            scores.append(score)

    word_overlap_score = statistics.mean(scores) if scores else 0

    del english_stopwords
    del sentiment_words

    return word_overlap_score

def get_stopwords():
    nltk_stopwords = set(stopwords.words('en'))
    #print(nltk_stopwords)
    sklearn_stopwords = stop_words.ENGLISH_STOP_WORDS

    all_stopwords = set()
    all_stopwords |= spacy_stopwords
    all_stopwords |= nltk_stopwords
    all_stopwords |= sklearn_stopwords

    return all_stopwords

sp='./style-words/yelp.txt'
def get_style_words():
    with open(file=sp,
              mode='r', encoding='ISO-8859-1') as sentiment_words_file:
        words = sentiment_words_file.readlines()
    words = set(word.strip() for word in words)
    return words




