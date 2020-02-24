import torch
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn

import sys
sys.path.append("..")
from util import trainer_util
from models.dissimilarity_model import DissimNet

class DissimilarityTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, config):
        cudnn.enabled = True
        self.config = config
        
        if config['gpu_ids'] != -1:
            self.gpu = 'cuda'
        else:
            self.gpu = 'cpu'
            
        self.diss_model = DissimNet().cuda(self.gpu)
        
        lr_config = config['optimizer']
        lr_options = lr_config['parameters']
        if lr_config['algorithm'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.diss_model.parameters(), lr=lr_options['lr'],
                                             weight_decay=lr_options['weight_decay'],)
        elif lr_config['algorithm'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.diss_model.parameters(),
                                              lr=lr_options['lr'],
                                              weight_decay=lr_options['weight_decay'],
                                              betas=(lr_options['beta1'], lr_options['beta2']))
        else:
            raise NotImplementedError
        
        self.old_lr = lr_options['lr']
        
        if config['training_strategy']['class_weight']:
            if not config['training_strategy']['class_weight_cityscapes']:
                label_path = os.path.join(config['train_dataloader']['dataset_args']['dataroot'], 'labels/')
                full_loader = trainer_util.loader(label_path, batch_size='all')
                print('Getting class weights for cross entropy loss. This might take some time.')
                class_weights = trainer_util.get_class_weights(full_loader, num_classes=2)
            else:
                class_weights = [1.46494611, 16.90304619]
            print('Using the following weights for each respective class [0,1]:', class_weights)
            self.criterion = nn.NLLLoss(weight=torch.FloatTensor(class_weights).to("cuda")).cuda(self.gpu)
        else:
            self.criterion = nn.NLLLoss(ignore_index=255).cuda(self.gpu)
        
        

    def run_model_one_step(self, original, synthesis, semantic, label):
        self.optimizer.zero_grad()
        predictions = self.diss_model(original, synthesis, semantic)
        model_loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1).cuda())
        model_loss.backward()
        self.optimizer.step()
        self.model_losses = model_loss
        self.generated = predictions
        return model_loss, predictions
        
    def run_validation(self, original, synthesis, semantic, label):
        predictions = self.diss_model(original, synthesis, semantic)
        model_loss = self.criterion(predictions, label.type(torch.LongTensor).squeeze(dim=1).cuda())
            
        return model_loss, predictions

    def get_latest_losses(self):
        return {**self.model_loss}

    def get_latest_generated(self):
        return self.generated

    def save(self, save_dir, epoch, name):
        if not os.path.isdir(os.path.join(save_dir, name)):
            os.mkdir(os.path.join(save_dir, name))
        
        save_filename = '%s_net_%s.pth' % (epoch, name)
        save_path = os.path.join(save_dir, name, save_filename)
        torch.save(self.diss_model.state_dict(), save_path)  # net.cpu() -> net

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.config['training_strategy']['niter']:
            lrd = self.config['optimizer']['parameters']['lr'] / self.config['training_strategy']['niter_decay']
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr


if __name__ == "__main__":
    import yaml
    
    config = '../configs/default_configuration.yaml'
    gpu_ids = 0

    with open(config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    config['gpu_ids'] = gpu_ids
    trainer = DissimilarityTrainer(config)
