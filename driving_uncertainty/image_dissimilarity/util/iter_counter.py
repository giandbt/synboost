import os
import time
import numpy as np

## Helper class that keeps track of training iterations
class IterationCounter():
    def __init__(self, config, dataset_size, batch_size, print_freq=100, display_freq=100):
        self.batch_size = batch_size
        self.config = config
        self.dataset_size = dataset_size
        
        self.first_epoch = 1
        self.total_epochs = config['training_strategy']['niter'] + config['training_strategy']['niter_decay']
        self.epoch_iter = 0  # iter number within each epoch
        
        self.total_steps_so_far = (self.first_epoch - 1) * dataset_size + self.epoch_iter
        
        self.print_freq = print_freq
        self.display_freq = display_freq
        
    
    # return the iterator of epochs for the training
    def training_epochs(self):
        return range(self.first_epoch, self.total_epochs + 1)
    
    def record_epoch_start(self, epoch):
        self.epoch_start_time = time.time()
        self.epoch_iter = 0
        self.last_iter_time = time.time()
        self.current_epoch = epoch
    
    def record_one_iteration(self):
        current_time = time.time()
        
        # the last remaining batch is dropped (see data/__init__.py),
        # so we can assume batch size is always batch_size
        self.time_per_iter = (current_time - self.last_iter_time) / self.batch_size
        self.last_iter_time = current_time
        self.total_steps_so_far += self.batch_size
        self.epoch_iter += self.batch_size
    
    def record_epoch_end(self):
        current_time = time.time()
        self.time_per_epoch = current_time - self.epoch_start_time
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (self.current_epoch, self.total_epochs, self.time_per_epoch))
    
    def needs_printing(self):
        return (self.total_steps_so_far % self.print_freq) < self.batch_size
    
    def needs_displaying(self):
        return (self.total_steps_so_far % self.display_freq) < self.batch_size





