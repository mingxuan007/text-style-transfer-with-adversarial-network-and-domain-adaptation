import tensorflow as tf

from network.nn import *
from models import BaseModel
from utils import *
import os
from tensorflow.contrib.rnn import GRUCell


from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops




class Model(BaseModel):
    def __init__(self, args, vocab):
        super().__init__(args, vocab)
        self.build_placeholder()
        self.build_model(args)

    def compute_batch_entropy(self, x):
        return tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=-x * tf.log(x + 1e-8), axis=1))

    def build_model(self, args):
        # auto-encoder
        with tf.variable_scope('encoder_decoder', reuse=tf.AUTO_REUSE):
            # word embedding
            embedding = tf.get_variable('embedding', initializer=self.word_init)
            # embedding = tf.get_variable('embedding', [self.vocab_size, self.dim_emb])
            enc_inputs = tf.nn.embedding_lookup(embedding, self.enc_inputs)
            dec_inputs = tf.nn.embedding_lookup(embedding, self.dec_inputs)
            enc_inputsz = tf.nn.embedding_lookup(embedding, self.sd_enc_inputs)
            dec_inputsz = tf.nn.embedding_lookup(embedding, self.sd_dec_inputs)
            with tf.variable_scope('projection'):
                # style information
                projection = {}
                projection['W'] = tf.get_variable('W', [self.dim_h, self.vocab_size])
                projection['b'] = tf.get_variable('b', [self.vocab_size])
            encoder = self.create_cell(self.dim_h, args.n_layers, self.dropout, 'encoder')
            decoder = self.create_cell(self.dim_h, args.n_layers, self.dropout, 'decoder')
           

            self.loss_rec, origin_info, transfer_info, real_sents = self.reconstruction(
                encoder, enc_inputs, self.labels,
                decoder, dec_inputs, self.targets, self.dec_mask, projection)
            self.loss_recz, _, _, _ = self.reconstructionz(
                encoder, enc_inputsz, self.sd_labels,
                decoder, dec_inputsz, self.sd_targets, self.sd_dec_mask, projection)
            fake_sents, soft_tsf_ids, self.rec_ids, self.tsf_ids = self.run_decoder(
                decoder, dec_inputs, embedding, projection, origin_info, transfer_info)

 
       
        # attention discriminator

        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            HIDDEN_SIZE = 256
            classifier_embedding = tf.get_variable('embedding', initializer=self.word_inity, trainable=True)
            # classifier_embedding = tf.get_variable('embedding', [self.vocab_size, self.dim_emb])
            # remove bos, use dec_inputs to avoid noises adding into enc_inputs
            real_sents = tf.nn.embedding_lookup(classifier_embedding, self.dec_inputs[:, 1:])
            fake_sents = tf.tensordot(soft_tsf_ids, classifier_embedding, [[2], [0]])
            fake_sents = fake_sents[:, :-1, :]  # make the dimension the same as real sents
            rnn_outputs, _ = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE),
                                    inputs=real_sents, sequence_length=self.enc_lens, dtype=tf.float32)
            rnn_outputsy, _ = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE),
                                     inputs=fake_sents, sequence_length=self.enc_lens, dtype=tf.float32)
            attention_output = self.attention(rnn_outputs)
            # mask the sequences
            drop = tf.nn.dropout(attention_output, self.dropout)
            attention_outputy = self.attention(rnn_outputsy)

            dropy = tf.nn.dropout(attention_outputy, self.dropout)
            with tf.name_scope('Fully_connected_layer'):
                W = tf.Variable(
                    tf.truncated_normal([HIDDEN_SIZE * 2, 1], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
                b = tf.Variable(tf.constant(0., shape=[1]))
                y_hat = tf.nn.xw_plus_b(drop, W, b)
                y_hat = tf.squeeze(y_hat)
                y_haty = tf.nn.xw_plus_b(dropy, W, b)
                y_haty = tf.squeeze(y_haty)
            with tf.name_scope('Metrics'):
                # Cross-entropy loss and optimizer initialization
                self.loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=self.labels))
                self.loss_g = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=y_haty, labels=1 - self.labels))

                # Accuracy metric
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), self.labels), tf.float32))
           

        #####   optimizer   #####
        

        theta_ed = retrive_var(['encoder_decoder'])
     
        theta_d = retrive_var(['discriminator'])
        theta_dz = retrive_var(['discriminatord'])
        theta_adv = get_var(['adver'])
    
        theta_eg = remove_var(theta_ed, theta_adv)
       
        loss_rec = self.loss_rec - self.style_adversary_entropy + self.loss_recz
        loss_recy = self.loss_rec - self.style_adversary_entropy + self.loss_g + self.loss_recz
        opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
        grady, _ = zip(*opt.compute_gradients(loss_recy, theta_eg))
        grady, _ = tf.clip_by_global_norm(grady, 30.0)

        self.optimize_adv = opt.minimize(self.style_adversary_loss, var_list=theta_adv)
        self.optimize_advg = opt.minimize(self.style_adversary_entropy, var_list=theta_eg)
        self.optimize_rec = opt.minimize(loss_rec, var_list=theta_eg)
        self.optimize_recy = opt.apply_gradients(zip(grady, theta_eg))
        self.optimize_d = opt.minimize(self.loss_d, var_list=theta_d)
        self.saver = tf.train.Saver(max_to_keep=1)

    def attention(self, inputs, attention_size=128):
    # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)


        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

        initializer = tf.random_normal_initializer(stddev=0.1)

        # Trainable parameters
        w_omega = tf.get_variable(name="w_omega", shape=[hidden_size, attention_size], initializer=initializer)
        b_omega = tf.get_variable(name="b_omega", shape=[attention_size], initializer=initializer)
        u_omega = tf.get_variable(name="u_omega", shape=[attention_size], initializer=initializer)

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
        return output

    def get_adv_style_p(self, embed):
        prob = tf.constant(1.0)
        style_adversary_mlp = tf.nn.dropout(
            x=tf.layers.dense(
                inputs=embed, units=256,
                activation=tf.nn.leaky_relu, name="style_adversary_mlp"),
            keep_prob=prob)

        style_adversary_prediction = tf.layers.dense(
            inputs=style_adversary_mlp, units=2,
            activation=tf.nn.softmax, name="style_adversary_prediction")
        return style_adversary_prediction




    def masked_attention(self, e):
        """Take softmax of e then apply enc_padding_mask and re-normalize"""
        attn_dist = nn_ops.softmax(e)
        print('4', attn_dist)  # take softmgpuidax. shape (batch_size, attn_length)
        attn_dist *= self.enc_mask  # apply mask
        masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
        return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

  


    def reconstructionz(self, encoder, enc_inputs, labels,
                        decoder, dec_inputs, targets, dec_mask, projection):
        labels = tf.reshape(labels, [-1, 1])
        batch_size = tf.shape(labels)[0]
  
        outputx, latent_vector = tf.nn.dynamic_rnn(encoder, enc_inputs,
                                                   scope='encoder', dtype=tf.float32)

        latent_vector = latent_vector[:, self.dim_y:]
       
        origin_info = tf.concat([self.linear(labels, self.dim_y,
                                             scope='output_styley'), latent_vector], 1)
        transfer_info = tf.concat([self.linear(1 - labels, self.dim_y,
                                               scope='output_styley', reuse=True), latent_vector], 1)

        hiddens, _ = tf.nn.dynamic_rnn(decoder, dec_inputs,
                                       initial_state=origin_info, scope='decoder')

        real_sents = tf.concat([tf.expand_dims(origin_info, 1), hiddens], 1)

        hiddens = tf.nn.dropout(hiddens, self.dropout)
        hiddens = tf.reshape(hiddens, [-1, self.dim_h])
        logits = tf.matmul(hiddens, projection['W']) + projection['b']

        rec_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(targets, [-1]), logits=logits)
        rec_loss *= tf.reshape(dec_mask, [-1])
        rec_loss = tf.reduce_sum(rec_loss) / tf.to_float(batch_size)

        return rec_loss, origin_info, transfer_info, real_sents

    def reconstruction(self, encoder, enc_inputs, labels,
                       decoder, dec_inputs, targets, dec_mask, projection):
        labels = tf.reshape(labels, [-1, 1])
        batch_size = tf.shape(labels)[0]

        outputx, latent_vector = tf.nn.dynamic_rnn(encoder, enc_inputs,
                                                   scope='encoder', dtype=tf.float32)

        latent_vector = latent_vector[:, self.dim_y:]   #get the initial content vector
        #remove irrelative style information in content vector
        with tf.variable_scope('adver', reuse=tf.AUTO_REUSE):
            style_adv_p = self.get_adv_style_p(latent_vector)
            self.style_adversary_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.one_labels, logits=style_adv_p,
                label_smoothing=0.1)
        self.style_adversary_entropy = self.compute_batch_entropy(style_adv_p)
        origin_info = tf.concat([self.linear(labels, self.dim_y,
                                             scope='output_style'), latent_vector], 1)
        transfer_info = tf.concat([self.linear(1 - labels, self.dim_y,
                                               scope='output_style', reuse=True), latent_vector], 1)
        self.origin_info = origin_info
        hiddens, _ = tf.nn.dynamic_rnn(decoder, dec_inputs,
                                       initial_state=origin_info, scope='decoder')
        self.hiddens = hiddens
        real_sents = tf.concat([tf.expand_dims(origin_info, 1), hiddens], 1)
        self.real_sents = real_sents
        hiddens = tf.nn.dropout(hiddens, self.dropout)
        hiddens = tf.reshape(hiddens, [-1, self.dim_h])
        logits = tf.matmul(hiddens, projection['W']) + projection['b']
        self.logits = logits
        self.targets = targets
        rec_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(targets, [-1]), logits=logits)
        rec_loss *= tf.reshape(dec_mask, [-1])
        rec_loss = tf.reduce_sum(rec_loss) / tf.to_float(batch_size)

        return rec_loss, origin_info, transfer_info, real_sents

   
#feed the information of batches
    def _make_feed_dict(self, batch, sd_batch=None, mode='train'):
        feed_dict = {}
        if mode == 'train':
            dropout = self.dropout_rate
        else:
            dropout = 1.0

        feed_dict[self.dropout] = dropout
        feed_dict[self.batch_len] = batch.batch_len
        feed_dict[self.enc_inputs] = batch.enc_batch
        feed_dict[self.dec_inputs] = batch.dec_batch
        feed_dict[self.labels] = batch.labels
        feed_dict[self.one_labels] = batch.one_labels
        feed_dict[self.targets] = batch.target_batch
        feed_dict[self.dec_mask] = batch.dec_padding_mask
        feed_dict[self.enc_mask] = batch.enc_padding_mask
        feed_dict[self.enc_lens] = batch.enc_lens
        if sd_batch != None:
            feed_dict[self.sd_batch_len] = sd_batch.batch_len
            feed_dict[self.sd_enc_inputs] = sd_batch.enc_batch
            feed_dict[self.sd_dec_inputs] = sd_batch.dec_batch
            feed_dict[self.sd_labels] = sd_batch.labels
            feed_dict[self.sd_one_labels] = sd_batch.one_labels
            feed_dict[self.sd_enc_lens] = sd_batch.enc_lens
            feed_dict[self.sd_targets] = sd_batch.target_batch
            feed_dict[self.sd_dec_mask] = sd_batch.dec_padding_mask
        return feed_dict

    def run_train_step(self, sess, batch, sd_batch, accumulator, epoch):
        feed_dict = self._make_feed_dict(batch, sd_batch)
        #pretrain
        if epoch <= 5:
            to_return = {'loss_rec': self.loss_rec,
                         'loss_recz': self.loss_recz,
                         'loss_style': self.style_adversary_loss,
                         'loss_g': self.loss_g,
                         'loss_d': self.loss_d,
                         'loss_en': self.style_adversary_entropy,
                         'accuracy': self.accuracy,
                         '2': self.optimize_rec,
                         '3': self.optimize_d,
                         '1': self.optimize_adv,
                         }
        #formal train
        else:
            to_return = {             
                '1': self.optimize_recy,
                '2': self.optimize_adv,
                '3': self.optimize_d,  
                'loss_rec': self.loss_rec,
                'loss_recz': self.loss_recz,              
                'loss_style': self.style_adversary_loss,
                'loss_g': self.loss_g,
                'loss_d': self.loss_d,
                'accuracy': self.accuracy,
                'loss_en': self.style_adversary_entropy}
        results1 = sess.run(to_return, feed_dict)
        results = {**results1}
        accumulator.add([results[name] for name in accumulator.names])
        return results1

    def run_eval_step(self, sess, batch):
        feed_dict = self._make_feed_dict(batch, mode='eval')

        to_return = {'rec_ids': self.rec_ids,
                     'tsf_ids': self.tsf_ids,
                     'loss_rec': self.loss_rec,

                     }

        return sess.run(to_return, feed_dict)

    def get_output_names(self):
        return ['loss_rec',
                'loss_recz',

                'loss_style',
                'loss_g',

                'loss_d',
                'loss_en']
