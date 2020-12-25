import random

from loader.Dataloader import Dataloader
#Dataloader for source and target
class MultiDataloader(Dataloader):
    def __init__(self, hps, vocab):
        self._vocab = vocab
        self._hps = hps

        # target dataset
        print('loading target dataset: %s' % self._hps.dataset)
        pos_queue, neg_queue = self.fill_example_queue(hps.target_train_path, mode = 'train')
        self.target_train_batch = self.create_batch(pos_queue, neg_queue, mode="train")
        random.shuffle(self.target_train_batch)
        target_len = len(self.target_train_batch)
        pos_queue, neg_queue = self.fill_example_queue(hps.target_test_path, mode = 'test')
        self.target_test_batch = self.create_batch(pos_queue, neg_queue, mode="test")

        # source dataset
        print('loading source dataset: %s' % self._hps.source_dataset)
        pos_queue, neg_queue = self.fill_example_queue(hps.source_train_path, mode = 'train')
        self.source_train_batch = self.create_batch(pos_queue, neg_queue, mode="train")
        source_len = len(self.source_train_batch)
        pos_queue, neg_queue = self.fill_example_queue(hps.source_test_path, mode = 'test')
        self.source_test_batch = self.create_batch(pos_queue, neg_queue, mode="test")

        # update checkpoint step
        checkpoint_step = int(max(source_len, target_len) / (hps.train_checkpoint_frequency*50)) * 50
        hps.train_checkpoint_step = max(50, checkpoint_step)


    def get_batches(self, domain='source', mode="train"):
        if domain == 'source':
            if mode == "train":
                random.shuffle(self.source_train_batch)
                return self.source_train_batch

            elif mode == 'test':
                return self.source_test_batch
            else:
                raise ValueError('Wrong mode name: %s.' % mode)
        elif domain == 'target':
            if mode == "train":
                random.shuffle(self.target_train_batch)
                return self.target_train_batch

            elif mode == 'test':
                return self.target_test_batch
            else:
                raise ValueError('Wrong mode name: %s.' % mode)
        else:
            raise ValueError('Wrong domain name: %s.' % domain)
