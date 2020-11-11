import numpy as np
import tensorflow as tf
import sys

from network.nn import leaky_relu, softsample_word, argmax_word


class BaseModel(object):
    def __init__(self, args, vocab):
        self.dim_emb = args.dim_emb
        self.vocab_size = vocab.size
        self.dim_y = args.dim_y
        self.dim_z = args.dim_z
        self.dim_h = self.dim_y + self.dim_z
        self.max_len = args.max_len
        self.dropout_rate = args.dropout_rate
        self.learning_rate = args.learning_rate
        self.rho = args.rho
        self.gamma = args.gamma_init

        self.pretrain_epochs = args.pretrain_epochs
        embedding_matrix = np.random.random_sample((self.vocab_size, self.dim_emb)) - 0.5
        embedding_matrixy = np.random.random_sample((self.vocab_size, 150)) - 0.5
        # initializer for word embeeding
        self.word_init = embedding_matrix.astype(np.float32)
        self.word_inity = embedding_matrixy.astype(np.float32)
        self.trunc_norm_init = tf.truncated_normal_initializer(stddev=1e-4)
        self.wei_init=tf.random_uniform_initializer(-0.001, 0.001)
    def build_placeholder(self):
        self.dropout = tf.placeholder(tf.float32,
            name='dropout')
        self.batch_len = tf.placeholder(tf.int32,
            name='batch_len')
        self.sd_batch_len = tf.placeholder(tf.int32,
                                        name='sd_batch_len')
        self.batch_size = tf.placeholder(tf.int32,
                                        name='batch_size')
        self.enc_inputs = tf.placeholder(tf.int32, [None, None],    #size * len
            name='enc_inputs')
        self.sd_enc_inputs = tf.placeholder(tf.int32, [None, None],  # size * len
                                         name='sd_enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, None],
            name='dec_inputs')
        self.sd_dec_inputs = tf.placeholder(tf.int32, [None, None],
                                         name='sd_dec_inputs')
        self.content_inputs = tf.placeholder(tf.int32, [None, None],
                                         name='content_inputs')
        self.targets = tf.placeholder(tf.int32, [None, None],
            name='targets')
        self.sd_targets = tf.placeholder(tf.int32, [None, None],
                                      name='sd_targets')
        self.dec_mask = tf.placeholder(tf.float32, [None, None],
            name='dec_mask')
        self.sd_dec_mask = tf.placeholder(tf.float32, [None, None],
                                       name='sd_dec_mask')
        self.enc_mask = tf.placeholder(tf.float32, [None, None],
                                       name='enc_mask')
        self.labels = tf.placeholder(tf.float32, [None],
            name='labels')
        self.sd_labels = tf.placeholder(tf.float32, [None],
                                     name='sd_labels')
        self.one_labels = tf.placeholder(tf.int32, [None,None],
                                     name='one_labels')
        self.sd_one_labels = tf.placeholder(tf.int32, [None, None],
                                         name='sd_one_labels')
        self.rev_labels = tf.placeholder(tf.int32, [None, None],
                                         name='rev_labels')
        self.enc_lens = tf.placeholder(tf.int32, [None],
            name='enc_lens')
        self.labels_d = tf.placeholder(tf.float32, [None],
                                       name='labels_d')
        self.sd_labels_d = tf.placeholder(tf.float32, [None],
                                          name='sd_labels_d')
        self.sd_enc_lens = tf.placeholder(tf.int32, [None],
                                       name='sd_enc_lens')
        self.content_lens = tf.placeholder(tf.int32, [None],
                                       name='content_lens')
        self.input_bow_representationc = tf.placeholder(
            dtype=tf.float32, shape=[None, None],
            name="input_bow_representations")
        self.nouns = tf.placeholder(
            dtype=tf.float32, shape=[None, None],
            name="nouns")
        self.input_bow_representation = tf.placeholder(
            dtype=tf.float32, shape=[None, None],
            name="input_bow_representationsy")
    def create_cell(self, dim, n_layers, dropout, scope=None):
        with tf.variable_scope(scope or "rnn"):
            cell = tf.nn.rnn_cell.GRUCell(dim)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                input_keep_prob=dropout)
            if n_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * n_layers,state_is_tuple=True)
        return cell

    def create_cell_with_dims(self, args, hidden_dim, input_dim, dropout, scope):
        cell = tf.nn.rnn_cell.GRUCell(hidden_dim)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)
        inputs = tf.placeholder(tf.float32, [args.batch_size, args.max_len, input_dim])
        with tf.variable_scope(scope):
            _, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        return cell
    #build the embedding of style labels
    def linear(self, inp, dim_out, scope, reuse=False):
        dim_in = inp.get_shape().as_list()[-1]
        print('in',dim_in)
        with tf.variable_scope(scope) as vs:
            if reuse:
                vs.reuse_variables()
            W = tf.get_variable('W', [dim_in, dim_out])
            b = tf.get_variable('b', [dim_out])
        return tf.matmul(inp, W) + b

    def rnn_decode(self, h, inp, length, cell, loop_func, scope):
        h_seq, output_ids = [], []

        with tf.variable_scope(scope):
                tf.get_variable_scope().reuse_variables()
                for t in range(length):
                    h_seq.append(tf.expand_dims(h, 1))
                    output, h = cell(inp, h)
                    inp, ids = loop_func(output)
                    output_ids.append(tf.expand_dims(ids, 1))
                h_seq.append(tf.expand_dims(h, 1))

        return tf.concat(h_seq, 1), tf.concat(output_ids, 1)

    def run_decoder(self, decoder, dec_inputs, embedding, projection, origin_info, transfer_info):
        go = dec_inputs[:,0,:]
        soft_func = softsample_word(self.dropout, projection['W'], projection['b'], embedding,
            self.gamma)
        hard_func = argmax_word(self.dropout, projection['W'], projection['b'], embedding)


        soft_tsf_hiddens, soft_tsf_ids, = self.rnn_decode(
            transfer_info, go, self.max_len, decoder, soft_func, scope='decoder')

        _, rec_ids = self.rnn_decode(
            origin_info, go, self.max_len, decoder, hard_func, scope='decoder')
        _, tsf_ids = self.rnn_decode(
            transfer_info, go, self.max_len, decoder, hard_func, scope='decoder')
        return soft_tsf_hiddens, soft_tsf_ids, rec_ids, tsf_ids

    

