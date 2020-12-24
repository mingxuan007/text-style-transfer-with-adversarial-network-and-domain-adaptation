import os
import sys
import pprint
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime as dt

#logger_name = "linguistic_style_transfer"

experiment_timestamp = dt.now().strftime("%Y%m%d%H%M%S")
save_directory = "./saved-models/{}".format(experiment_timestamp)

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    # data path
    argparser.add_argument('--dataDir',
            type=str,
            default='./data')
    argparser.add_argument('--dataset',
            type=str,
            default='yelp',
            help='if doman_adapt enable, dataset means target dataset')
    argparser.add_argument('--style_words_path',
            type=str,
            default='./style_words/yelp.txt')
    argparser.add_argument('--modelDir',
            type=str,
            default='./save_model')
    argparser.add_argument('--logDir',
            type=str,
            default='')
    #glove100d   path
    argparser.add_argument('--validation_embeddings_file_path',
            type=str,
            default='')
    argparser.add_argument('--style_file_path',
            type=str,
            default='./save_model/style')
    argparser.add_argument('--domain_file_path',
            type=str,
            default='./save_model/domain')
    # general model setting
    argparser.add_argument('--learning_rate',
            type=float,
            default=0.0005)
    argparser.add_argument('--batch_size',
            type=int,
            default=128)
    argparser.add_argument('--pretrain_epochs',
            type=int,
            default=5,
            help='max pretrain epoch for LM.')
    argparser.add_argument('--max_epochs',
            type=int,
            default=20)
    argparser.add_argument('--max_len',
            type=int,
            default=16,
            help='the max length of sequence+one mark')



   
    argparser.add_argument('--rho',                 # loss_rec + rho * loss_adv
            type=float,
            default=1)
    argparser.add_argument('--gamma_init',          # softmax(logit / gamma)
            type=float,
            default=0.1)
    argparser.add_argument('--gamma_decay',
            type=float,
            default=1)
    argparser.add_argument('--gamma_min',
            type=float,
            default=0.1)
    argparser.add_argument('--beam',
            type=int,
            default=1)
    argparser.add_argument('--dropout_rate',
            type=float,
            default=0.8)
    argparser.add_argument('--n_layers',
            type=int,
            default=1)
    argparser.add_argument('--dim_y',
            type=int,
            default=200)
    #500 represent the dim of content information in latent representation
    argparser.add_argument('--dim_z',
            type=int,
            default=500)
    argparser.add_argument('--dim_emb',
            type=int,
            default=100)

    # training config
    argparser.add_argument('--suffix',
            type=str,
            default='')
    argparser.add_argument('--load_model',
            action='store_true',
            help='whether load the model for test')
    argparser.add_argument('--save_model',
            default=True,
            action='store_true',
            help='whether save the model for test')
    argparser.add_argument('--train_checkpoint_frequency',
            type=int,
            default=4,
            help='how many checkpoints in one training epoch')
    argparser.add_argument('--training_portion',
            type=float,
            default=1.0)
    argparser.add_argument('--source_training_portion',
            type=float,
            default=1.0)

    # Multi-dataset support
    argparser.add_argument('--domain_adapt',
            action='store_true',
            default=True)
    argparser.add_argument('--source_dataset',
            type=str,
            default='imdb')
    argparser.add_argument('--dim_d',
            type=int,
            default=50,
            help='The dimension of domain vector.')
    argparser.add_argument('--alpha',
            type=float,
            default=0.5,
            help='The weight of domain loss.')

  


    args = argparser.parse_args()


    # update data path according to single dataset or multiple dataset



    update_domain_adapt_datapath(args)
    # update batch size if using parallel training
    if 'para' in args.dataset:
        args.batch_size = int(args.batch_size/2)

    # update output path
    if not args.logDir:

        args.logDir = 'logs'

    log_dir = Path(args.logDir)

    if not log_dir.exists():
        print('=> creating {}'.format(log_dir))
        log_dir.mkdir(parents = True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')        
    log_file = '{}.log'.format(time_str)


    final_log_file = log_dir / log_file

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)


    logger.info('------------------------------------------------')
    logger.info(pprint.pformat(args))
    logger.info('------------------------------------------------')

    return args

def update_domain_adapt_datapath(args):
    # update data path
    args.dataDir = os.path.join(args.dataDir, 'data')
    # target_data
    target_data_root = os.path.join(args.dataDir, args.dataset)
    args.target_train_path = os.path.join(target_data_root, 'train')
    args.train_path = os.path.join(target_data_root, 'train')
    args.target_valid_path = os.path.join(target_data_root, 'valid')
    args.target_test_path = os.path.join(target_data_root, 'test')
    args.test_path = os.path.join(target_data_root, 'test')
    # the vocabulary used for classifier evaluation
    args.target_vocab = os.path.join(target_data_root, 'vocab')
    args.vocab = os.path.join(target_data_root, 'vocab')
    # source data
    source_data_root = os.path.join(args.dataDir, args.source_dataset)
    args.source_train_path = os.path.join(source_data_root, 'train')
    args.source_train_pathz = os.path.join(source_data_root, 'train')
    args.source_valid_path = os.path.join(source_data_root, 'valid')
    args.source_test_path = os.path.join(source_data_root, 'test')
    # the vocabulary used for classifier evaluation
    args.source_vocab = os.path.join(source_data_root, 'vocab')

    # save the togather vocab in common root 'data/multi_vocab'
    args.multi_vocab = os.path.join(
        args.dataDir, '_'.join([args.source_dataset, args.dataset, 'multi_vocab']))

    # update output path
    args.modelDir = os.path.join(args.modelDir, 'save_model')
    args.classifier_path = os.path.join(args.modelDir, 'classifier', args.dataset)
    args.lm_path = os.path.join(args.modelDir, 'lm', args.dataset)
    args.styler_path = os.path.join(args.modelDir, args.network, args.dataset)

    return args
