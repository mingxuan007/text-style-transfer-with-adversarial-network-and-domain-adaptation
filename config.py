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
save_directory = "/share/nishome/20010431_1/XXX/linguistic_style_transfer_model/saved-models/{}".format(experiment_timestamp)

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    # data path
    argparser.add_argument('--dataDir',
            type=str,
            default='/share/nishome/20010431_1/Desktop/DASTy')
    argparser.add_argument('--dataset',
            type=str,
            default='yelp',
            help='if doman_adapt enable, dataset means target dataset')
    argparser.add_argument('--modelDir',
            type=str,
            default='/share/nishome/20010431_1/Desktop/DASTy/save')
    argparser.add_argument('--logDir',
            type=str,
            default='')

    # general model setting
    argparser.add_argument('--learning_rate',
            type=float,
            default=0.0005)
    argparser.add_argument('--batch_size',
            type=int,
            default=128)
    argparser.add_argument('--pretrain_epochs',
            type=int,
            default=10,
            help='max pretrain epoch for LM.')
    argparser.add_argument('--max_epochs',
            type=int,
            default=20)
    argparser.add_argument('--max_len',
            type=int,
            default=16,
            help='the max length of sequence')
    argparser.add_argument('--noise_word',
            default=False,
            help='whether add noise in enc batch.')
    argparser.add_argument('--trim_padding',
            action='store_true',
            default=False)
    argparser.add_argument('--order_data',
            action='store_true',
            help='whether order the data according the length in the dataset.')

  
    argparser.add_argument('--confidence',
            type=float,
            default=0.8,
            help='The classification confidence used to filter the data')

   
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
            default='filter_imdb')
    argparser.add_argument('--dim_d',
            type=int,
            default=50,
            help='The dimension of domain vector.')
    argparser.add_argument('--alpha',
            type=float,
            default=0.5,
            help='The weight of domain loss.')

  


    args = argparser.parse_args()
    # check whether use online annotated dataset from human
    if args.dataset in ['yelp', 'amazon']:
        args.online_test = True

    # update data path according to single dataset or multiple dataset

    args.dataDir = os.path.join(args.dataDir, 'data')
    data_root = os.path.join(args.dataDir, args.dataset)
    args.train_path = os.path.join(data_root, 'train')

    args.test_path = os.path.join(data_root, 'test')
    args.vocab = os.path.join(data_root, 'vocab')
    args.source_vocab = os.path.join(data_root, 'source_vocab')
    
    args.target_train_path = os.path.join(data_root, 'train')

    args.target_test_path = os.path.join(data_root, 'test')
    source_data_root = os.path.join(args.dataDir, args.source_dataset)
    args.source_train_path = os.path.join(source_data_root, 'trainy')
    args.source_train_pathz = os.path.join(source_data_root, 'train')

    args.source_test_path = os.path.join(source_data_root, 'test')
    # update output path
    args.modelDir = os.path.join(args.modelDir, 'save_model')
    args.classifier_path = os.path.join(args.modelDir, 'classifier', args.dataset)
    args.lm_path = os.path.join(args.modelDir, 'lm', args.dataset)
    args.styler_path = os.path.join(args.modelDir, args.network, args.dataset)

    update_domain_adapt_datapath(args)
    # update batch size if using parallel training
    if 'para' in args.dataset:
        args.batch_size = int(args.batch_size/2)

    # update output path
    if not args.logDir:
        # if not in philly enviroment
        args.logDir = 'logs'
        args.logDir = os.path.join(args.logDir, 'XXX', args.suffix)
    log_dir = Path(args.logDir)

    if not log_dir.exists():
        print('=> creating {}'.format(log_dir))
        log_dir.mkdir(parents = True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')        
    log_file = '{}_{}_{}.log'.format('resAAw_lossgy', args.suffix, time_str)
    # update the suffix for tensorboard file name
    args.suffix = '{}_{}_{}'.format('resAAw_lossgy', args.suffix, time_str)

    final_log_file = log_dir / log_file
    print('12345qw',final_log_file)
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
    args.mvocabz = os.path.join(
        args.dataDir, '_'.join([args.source_dataset, args.dataset, 'mvocabz']))
    # update output path
    args.modelDir = os.path.join(args.modelDir, 'save_model')
    args.classifier_path = os.path.join(args.modelDir, 'classifier', args.dataset)
    args.lm_path = os.path.join(args.modelDir, 'lm', args.dataset)
    args.styler_path = os.path.join(args.modelDir, args.network, args.dataset)

    return args
