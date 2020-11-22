import sys
import config
import argparse
import datetime
import json
import numpy as np
import os
import tensorflow as tf


from TextCNN import TextCNN


logger = None


def train_classifier_model(options):
    # Load data
    logger.info("Loading data...")
    f=open(options['text_file_path'])
    text=[]
    y=[]
    for i in f.readlines():
        te=json.loads(i.rstrip())
        text.append(te["review"])
        y.append(te["score"])
    text_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='!"#$%&()*+-./:=?@[\\]^_`{|}~\t\n')
    predefined_word={"<pad>": 0, "<go>": 1, "<eos>": 2} 
    for index, word in enumerate(text_tokenizer.word_index):
         new_index = index + 3
         word_index[word] = new_index
    text_tokenizer.word_index = word_index
    sequences = text_tokenizer.texts_to_sequences(text)
    x = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=config.max_sequence_length, padding='post',
        truncating='post', value=word_index[global_config.eos_token])


    x = np.asarray(x)

    
    
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(0.01 * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled


    logger.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    # Training
    sess = tf_session_helper.get_tensorflow_session()
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=options['vocab_size'],
            embedding_size=128,
            filter_sizes=list(map(int, [3, 4, 5])),
            num_filters=128,
            l2_reg_lambda=0.0)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.RMSPropOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)




        

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

        # Write vocabulary
        with open(global_config.classifier_vocab_save_path, 'w') as json_file:
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
        batches = data_processor.batch_iter(
            list(zip(x_train, y_train)), mconf.batch_size, options['training_epochs'])
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 100 == 0:
                logger.info("\nEvaluation:")
                accuracy=dev_step(x_dev, y_dev, writer=dev_summary_writer)
                logger.info("")
            if current_step % 100 == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                logger.info("Saved model checkpoint to {}\n".format(path))
                if current_step>25000:
                    if accuracy > 0.978:
                        break

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file-path", type=str, default='./data/yelp/train.txt')
    
    
    parser.add_argument("--training-epochs", type=int, default=20)
    parser.add_argument("--logging-level", type=str, default="INFO")

    options = vars(parser.parse_args(args=argv))
    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, options['logging_level'])

    os.makedirs(global_config.classifier_save_directory)

    train_classifier_model(options)

    logger.info("Training Complete!")




if __name__ == "__main__":
    main(sys.argv[1:])
