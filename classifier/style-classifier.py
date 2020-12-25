import sys
import config
import argparse
import datetime
import json
import numpy as np
import os
import tensorflow as tf
import logging
from config import load_arguments
os.environ["CUDA_VISIBLE_DEVICES"]='3'
from TextCNN import TextCNN
args=load_arguments(1)

logger = None


def setup_custom_logger(name, log_level):
    formatter = logging.Formatter(
        fmt="%(asctime)s: %(message)s",
        datefmt="%m-%dT%H:%M:%S")

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.addHandler(handler)

    return logger

def train_classifier_model(options):
    # Load data
    logger.info("Loading data...")
    f=open(options['text_file_path'])
    text=[]
    y=[]
    for i in f.readlines():
        te=json.loads(i.rstrip())
        text.append(te["review"])
        if te["score"] == 0:
           y.append([0,1])
        else:
           y.append([1,0])
    text_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='!"#$%&()*+-./:=?@[\\]^_`{|}~\t\n')
    text_tokenizer.fit_on_texts(text)
    word_index={"<pad>": 0, "<go>": 1, "<eos>": 2}
    for index, word in enumerate(text_tokenizer.word_index):
         new_index = index + 3
         word_index[word] = new_index
    text_tokenizer.word_index = word_index
    sequences = text_tokenizer.texts_to_sequences(text)
    x = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=args.max_len-1, padding='post',
        truncating='post', value=word_index["<eos>"])


    x = np.array(x)
    print("zxcas",x.shape)

    
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    y=np.array(y)
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(0.01 * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled
    embedding_index = {}
    print(options['embedding'])
    f = open(options['embedding'], 'r', encoding='utf-8')
    for line in f:
        values = line.split()
        
        word = values[0]

        coef = np.asarray(values[1:], dtype='float32')
        embedding_index[''.join(word)] = coef
    f.close()

    vocab_size = len(word_index)
    embedding_matrix = np.zeros([vocab_size, 300])
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    embedding_matrix = tf.convert_to_tensor(embedding_matrix, dtype=tf.float32)

    logger.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    gpu_options = tf.GPUOptions(allow_growth=True)
    config_proto = tf.ConfigProto(
        log_device_placement=False, allow_soft_placement=True,
        gpu_options=gpu_options)
    sess = tf.Session(config=config_proto)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=vocab_size,
            embedding_size=300,
            filter_sizes=list(map(int, [3, 4, 5])),
            num_filters=128,embedding=embedding_matrix,
            l2_reg_lambda=0.0)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.RMSPropOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        out_dir = options["model_file_path"]
        logger.info("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        # train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)



        


 
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

        # Write vocabulary
        with open(options["vocab_file_path"], 'w') as json_file:
            json.dump(word_index, json_file)
            logger.info("Saved vocabulary")

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 0.5
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            logger.info("step {}: loss {:g}, acc {:g}".format(step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            logger.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return accuracy
        # Generate batches
        def batch_iter(data, batch_size, num_epochs, shuffle=True):
            """
            Generates a batch iterator for a dataset.
            """
            data = np.array(data)
            data_size = len(data)
            num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
            for epoch in range(num_epochs):
                # Shuffle the data at each epoch
                if shuffle:
                    shuffle_indices = np.random.permutation(np.arange(data_size))
                    shuffled_data = data[shuffle_indices]
                else:
                    shuffled_data = data
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)
                    yield shuffled_data[start_index:end_index]
        batches = batch_iter(
            list(zip(x_train, y_train)), args.batch_size, options['training_epochs'])
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            x_batch=np.array(x_batch)
            y_batch=np.array(y_batch)

            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 100 == 0:
                logger.info("\nEvaluation:")
                accuracy=dev_step(x_dev, y_dev, writer=dev_summary_writer)
                logger.info("")
            if current_step % 100 == 0:
                path = saver.save(sess, options["model_file_path"]+"/checkpoints/style", global_step=current_step)
                logger.info("Saved model checkpoint to {}\n".format(path))
                if current_step>10000:
                    if accuracy > 0.96:
                        break

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file-path", type=str, default='../data/yelp/train/ntrain.txt')
    #
    parser.add_argument("--embedding", type=str, default='/share/nishome/20010431_1/Downloads/glove.6B.300d.txt')
    parser.add_argument("--training-epochs", type=int, default=20)
    parser.add_argument("--logging-level", type=str, default="INFO")
    parser.add_argument("--model-file-path",type=str,default='../save_model/style')
    parser.add_argument("--vocab-file-path",type=str,default='../save_model/style/vocab.txt')
    options = vars(parser.parse_args(args=argv))
    global logger
    logger = setup_custom_logger("style-classifier", options['logging_level'])
    if not os.path.exists(options["model_file_path"]):
       os.makedirs(options["model_file_path"])


    train_classifier_model(options)

    logger.info("Training Complete!")




if __name__ == "__main__":
    main(sys.argv[1:])
