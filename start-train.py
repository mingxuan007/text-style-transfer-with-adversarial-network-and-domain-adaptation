import os
import sys
import time
import random
import logging
from utils import *
import network
import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction





from vocab import Vocabulary, build_vocab
from config import load_arguments
from dataloader.style_dataloader import StyleDataloader
from dataloader.multi_style_dataloader import MultiStyleDataloader

os.environ["CUDA_VISIBLE_DEVICES"]='0'
smoothie = SmoothingFunction().method4
import json
logger = logging.getLogger(__name__)
classifier_saved_model_path=''
domain_classifier_model=''

validation_embeddings_file_path='/share/nishome/20010431_1/Documents/glove.6B.100d.txt'
def evaluation(sess, vocab, batches, model, path,output_path, epoch ):
    transfer_acc = 0
    origin_acc = 0
    total = 0
    ref = []
    ori_ref = []
    hypo = []
    origin = []
    transfer = []
    reconstruction = []
    ltsf=[]
    #accumulator = Accumulator(len(batches), model.get_output_names())
    a=np.array([])
    print('aw')
    for batch in batches:
        
        results = model.run_eval_step(sess, batch)
  
        #reconstructed data
        rec = [[vocab.id2word(i) for i in sent] for sent in results['rec_ids']]
        rec, _ = strip_eos(rec)
        rec = [' '.join(i) for i in rec]
        #transferred data
        tsf = [[vocab.id2word(i) for i in sent] for sent in results['tsf_ids']]
        tsf, lengths = strip_eos(tsf)
        tsf = [' '.join(i) for i in tsf]
        reconstruction.extend(rec)
        transfer.extend(tsf)
        hypo.extend(tsf)
        origin.extend(batch.original_reviews)

        for x in batch.original_reviews:
            ori_ref.append(x.split())
        for x in tsf:
            ltsf.append(x.split())
        a=np.concatenate((a,batch.labels),axis=0)
    # evaluate 
    [style_transfer_score, confusion_matrix] = style_transfer.get_style_transfer_score3(
        classifier_saved_model_path, transfer, a)
    [domain_acc, confusion_matrix] = style_transfer.get_style_transfer_score1(
        domain_classifier_model, transfer, 1, None)
    glove_model = load_glove_model(validation_embeddings_file_path)
    content_preservation_score = get_content_preservation_score(
        ori_ref, ltsf, glove_model)
    word_overlap_score = get_word_overlap_score(
        ori_ref, ltsf)
    ori_b = []
    for i in ori_ref:
        ori_b.append([i])
   

    bleu = corpus_bleu(ori_b, ltsf,weights=[0.34,0.33,0.33], smoothing_function=smoothie)
    logger.info("Bleu score: %.4f" % bleu)
    #upload score
    with open(path, 'a+') as validation_scores_file:
        validation_record = {
            "epoch": epoch,
            "domain_acc": domain_acc,
            "style-transfer": style_transfer_score,
            "content-preservation": content_preservation_score,
            "word-overlap": word_overlap_score,
            'bleu': bleu

        }
        validation_scores_file.write(json.dumps(validation_record) + "\n")

    write_output_v0(origin, transfer, reconstruction, output_path,epoch)

    return style_transfer_score, bleu


def create_model(sess, args, vocab):
    model = eval('network' + '.Model')(args, vocab)


    """logger.info('-----Loading styler model from: -----')
    model.saver.restore(sess, '/share/nishome/20010431_1/Desktop/DASTy/save/save_model/NNN/5')"""

    logger.info('-----Creating styler model with fresh parameters.-----')
    sess.run(tf.global_variables_initializer())


    if not os.path.exists(args.styler_path):
            os.makedirs(args.styler_path)
    return model
from utils import bow_len
if __name__ == '__main__':
    config = tf.ConfigProto()


   
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        args = load_arguments()

        if not os.path.isfile(args.vocab):
            build_vocab(args.train_path, args.vocab)
\
        vocab = Vocabulary(args.vocab)
        logger.info('vocabulary size: %d' % vocab.size)
    

        
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        write_dict = {
        'writer': tf.summary.FileWriter(logdir=tensorboard_dir, filename_suffix=args.suffix),
        'step': 0
        }
        
        # load data
        loader = MultiStyleDataloader(args, vocab)
      
        # create a folder for data samples
        output_path = os.path.join(args.logDir, args.dataset)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

  
        batches = loader.get_batches(domain='target', mode='train')
        sd_batches=loader.get_batches(domain='source', mode='train')


        # create style transfer model
        model = create_model(sess, args, vocab)
       
        #create folder to save 
        path=os.path.join(args.modelDir,'train_1')
        if not os.path.exists(path):
            os.makedirs(path)
        start_time = time.time()
        step = 0
        accumulator = Accumulator(args.train_checkpoint_step, model.get_output_names())
        learning_rate = args.learning_rate
    
   
      
        path1 = os.path.join(path, 'score.txt')
        path1y = os.path.join(path, 'test')
        
       
        acc_cut = 0
        gamma = args.gamma_init
        tes_batches = loader.get_batches(domain='target',mode='test')

        for epoch in range(1, 45):
            logger.info('--------------------epoch %d--------------------' % epoch)
            logger.info('learning_rate: %.4f  gamma: %.4f' % (learning_rate, gamma))


            total_batch = max(len(sd_batches), len(batches))
            
            for i in range(total_batch):
                result1=model.run_train_step(sess,batches[i % len(batches)], sd_batches[i % len(sd_batches)], accumulator,
                                     epoch)
                # train 300 step (mini-batch) to show the obejective loss values
                if step % 300 == 0:
               
                    accumulator.output('step %d, time %.0fs,'
                        % (step, time.time() - start_time),  'train')
                    accumulator.clear()
            #after pre-train, evaluation model        
            if epoch>5:
                acc, bleu = evaluation(sess, vocab, tes_batches, model,
                path1, path1y,  epoch)

