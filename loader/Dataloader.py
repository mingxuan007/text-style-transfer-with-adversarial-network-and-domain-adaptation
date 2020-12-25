import random
import copy
import glob
import codecs
import json
import numpy as np
import tensorflow as tf
import os

# for every sentence properties
class Example(object):
   
    def __init__(self, review, ref, label, vocab, hps):
       
        self.hps = hps

        # special tokens
        self.start_decoding = vocab.word2id('<go>')
        self.stop_decoding = vocab.word2id('<eos>')
        self.unk = vocab.word2id('<unk>')
        self.pad = vocab.word2id('<pad>')

        article_words = review.split()
        self.reference = ref
        self.original_review = review
        # self.original_review = ' '.join(article_words[:hps.max_len])
        self.original_len = len(article_words)

        # trunct to hps.max_len - 1, needs one extra token for <START> and <END>
        if len(article_words) > hps.max_len-1:
            article_words = article_words[:hps.max_len-1]
        # can see in vocab.py
        abs_ids=[]
        for w in article_words:
            abs_ids.append(vocab.word2id(w))
        #abs_ids = [vocab.word2id(w) if vocab.word2id(w) !=1 for w in article_words]
        self.label = int(label)

        # Get input sequence and target sequence
        self.enc_input = copy.copy(abs_ids)
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(copy.copy(abs_ids))
        self.dec_len = len(self.target)
    # append special tokens 
    def get_dec_inp_targ_seqs(self, sequence):

        inp = [self.start_decoding] + sequence[:]
        target = sequence[:] + [self.stop_decoding]
        assert len(inp) == len(target)
        return inp, target

    def pad_encoder_decoder_input(self, batch_len):
  
        # enc_input doesn't need <START> or <END>
        if len(self.enc_input) < batch_len - 1:
            padding = [self.pad] * (batch_len-1 - len(self.enc_input))
            self.enc_input.extend(padding)

        # pad dec_input and target
        if len(self.dec_input) < batch_len:
            padding = [self.pad] * (batch_len - len(self.dec_input))
            self.dec_input.extend(padding)
            self.target.extend(padding)


# for a mini-batch
class Batch(object):
    """Class representing a minibatch of train/val/test examples for text summarization."""

    def __init__(self, example_list, hps, vocab):
        """Turns the example_list into a Batch object.

        Args:
        example_list: List of Example objects
        hps: hyperparameters
        vocab: Vocabulary object
        """
        # process the encoder and decoder sequence
        self.init_encoder_decoder_seq(example_list, hps)  # initialize the input to the encoder'''
        # process original review and label
        self.original_reviews = [ex.original_review for ex in example_list] # store the original strings
        self.references = [ex.reference for ex in example_list] # store the reference

    def init_encoder_decoder_seq(self, example_list, hps):
        # Encoder seq
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.


        batch_len = hps.max_len
        # the main length is 5
        batch_len = max(batch_len, 5)
        batch_size = len(example_list)

        for ex in example_list:
            ex.pad_encoder_decoder_input(batch_len)

        self.enc_batch = np.zeros((batch_size, batch_len-1), dtype=np.int32)
        self.labels = np.zeros((batch_size), dtype=np.int32)
        self.enc_lens = np.zeros((batch_size), dtype=np.int32)
        # self.reward = np.zeros((batch_size), dtype=np.float32)
        self.one_labels = np.zeros((batch_size,2), dtype=np.int32)
        self.rev_labels = np.ones((batch_size, 2), dtype=np.int32)
        # Decoder seq
        self.dec_batch = np.zeros((batch_size, batch_len), dtype=np.int32)
        self.target_batch = np.zeros((batch_size, batch_len), dtype=np.int32)
        self.dec_padding_mask = np.zeros((batch_size, batch_len), dtype=np.float32)
        self.enc_padding_mask = np.zeros((batch_size, batch_len-1), dtype=np.float32)
        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            # encoder seq
            self.enc_batch[i, :] = ex.enc_input[:]
            self.labels[i] = ex.label
            self.one_labels[i][ex.label]=1
            self.rev_labels[i][ex.label] = 0
            self.enc_lens[i] = ex.original_len
            # decoder seq
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_padding_mask[i][:ex.dec_len] = 1.0
            self.enc_padding_mask[i][:ex.dec_len] = 1.0
            # self.reward[i] = ex.reward
        # the length of decoding batch
        self.batch_len = batch_len


class Dataloader(object):
    def __init__(self, hps, vocab):
        self._vocab = vocab
        self._hps = hps

        pos_queue, neg_queue = self.fill_example_queue(hps.train_path, mode='train')
        self.train_batch = self.create_batch(pos_queue, neg_queue, mode="train")
        random.shuffle(self.train_batch)
        portion = int(len(self.train_batch) * hps.training_portion)
        self.train_batch = self.train_batch[:portion]
        # update training checkpoint step
        checkpoint_step = int(portion / (hps.train_checkpoint_frequency*50)) * 50
        hps.train_checkpoint_step = max(50, checkpoint_step)



        pos_queue, neg_queue = self.fill_example_queue(hps.test_path, mode='test')
        self.test_batch = self.create_batch(pos_queue, neg_queue, mode="test")

    def create_batch(self, pos_queue, neg_queue, mode="train"):
        all_batch = []
        #assert len(pos_queue) == len(neg_queue)
        if mode=="train":
            begin = list(range(0, len(pos_queue), self._hps.batch_size))
            end = begin[1:] + [len(pos_queue)]

            for i, j in zip(begin, end):
                pos_batch = pos_queue[i : j]
                neg_batch = neg_queue[i : j]
                all_batch.append(Batch(neg_batch + pos_batch, self._hps, self._vocab))
        else:
            batch=pos_queue+neg_queue
            begin = list(range(0, len(batch), self._hps.batch_size))
            end = begin[1:] + [len(batch)]
            for i, j in zip(begin, end):
                nbatch = batch[i: j]

                all_batch.append(Batch(nbatch, self._hps, self._vocab))
        return all_batch

    def get_batches(self, mode="train"):
        if mode == "train":
            #random.shuffle(self.train_batch)
            return self.train_batch

        elif mode == 'test':
            return self.test_batch
        else:
            raise ValueError('Wrong batch number %s.' % mode)

    def fill_example_queue(self, data_path, mode="valid"):

        positive_queue =[]
        negative_queue =[]

        filelist = glob.glob(os.path.join(data_path, '*.txt'))  # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' % data_path)  # check filelist isn't empty

        for f in filelist:
            reader = codecs.open(f, 'r', 'utf-8')
            while True:
                string_ = reader.readline()
                if not string_: break
                dict_example = json.loads(string_)
                review = dict_example["review"]
                score = 1 if dict_example["score"] > 0 else 0

                if mode == 'test':
                    example = Example(review, review, score, self._vocab, self._hps)
                else:
                    example = Example(review, review, score, self._vocab, self._hps)
                if score == 1:
                    positive_queue.append(example)
                elif score == 0:
                    negative_queue.append(example)
                else:
                    raise ValueError('The score %d is not 0 or 1.' % score)

        print('%s file has %d positive unique sentences.' % (mode, len(positive_queue)))
        print('%s file has %d negative unique sentences.' % (mode, len(negative_queue)))
        print('%s file has %d total unique sentences.' % (mode, len(negative_queue) + len(positive_queue)))

        # keep positive and negative balance in training
        if mode == 'train':


            while True:
                if len(positive_queue) > len(negative_queue):
                    negative_queue.extend(copy.deepcopy(negative_queue[:len(positive_queue) - len(negative_queue)]))
                elif len(positive_queue) < len(negative_queue):
                    positive_queue.extend(copy.deepcopy(positive_queue[:len(negative_queue) - len(positive_queue)]))
                if len(positive_queue) == len(negative_queue):
                    break

        # sort the data according to the length
        
        positive_queue = sorted(positive_queue, key=lambda x: x.original_len)
        negative_queue = sorted(negative_queue, key=lambda x: x.original_len)

        return positive_queue, negative_queue
