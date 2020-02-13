from models.dissimilarity_model import DissimNet
import torch
import os
import torch.nn as nn

import sys
sys.path.append("..")
from util.cityscapes_dataset import trainer_util

class DissimilarityTrainer():
    #TODO (Giancarlo): Complete this class. Also add weights for loss from https://github.com/iArunava/ENet-Real-Time-Semantic-Segmentation/blob/master/train.py
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, config):
        self.config = config
        self.diss_model = DissimNet()
        
        lr_config = config['optimizer']
        lr_options = lr_config['parameters']
        if lr_config['algorithm'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.diss_model.parameters(), lr=lr_options['lr'])
        elif lr_config['algorithm'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.diss_model.parameters(),
                                              lr=lr_options['lr'], betas=(lr_options['beta1'], lr_options['beta2']))
        else:
            raise NotImplementedError
        
        self.old_lr = lr_options['lr']
        
        segmented_path = os.path.join(config['train_dataloader']['dataset_args']['dataroot'], 'semantic')
        full_loader = trainer_util.loader(segmented_path, batch_size='all')
        class_weights = trainer_util.get_class_weights(full_loader, num_classes=19)

        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to('gpu'))
        
        
    # TODO All these functions

    def run_model_one_step(self, data):
        self.optimizer.zero_grad()
        g_losses, generated = self.diss_model(data)
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def save(self, epoch):
        self.diss_model.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
