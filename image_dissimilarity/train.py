import argparse
import yaml
import torch.backends.cudnn as cudnn
import torch
from PIL import Image
import numpy as np

from trainers.dissimilarity_trainer import DissimilarityTrainer
from util import trainer_util
from util.iter_counter import IterationCounter

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
opts = parser.parse_args()
cudnn.benchmark = True

# Load experiment setting
with open(opts.config, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
    
# get experiment information
exp_name = config['experiment_name']
save_fdr = config['save_folder']

# Activate GPUs
config['gpu_ids'] = opts.gpu_ids
gpu_info = trainer_util.activate_gpus(config)

# Get data loaders
cfg_train_loader = config['train_dataloader']
cfg_val_loader = config['val_dataloader']
train_loader = trainer_util.get_dataloader(cfg_train_loader['dataset_args'], cfg_train_loader['dataloader_args'])
val_loader = trainer_util.get_dataloader(cfg_val_loader['dataset_args'], cfg_val_loader['dataloader_args'])

# create trainer for our model
trainer = DissimilarityTrainer(config)

# create tool for counting iterations
batch_size = config['train_dataloader']['dataloader_args']['batch_size']
iter_counter = IterationCounter(config, len(train_loader), batch_size)

# create tool for visualization
# TODO (Giancarlo): Need to add visualization tool

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    train_loss = 0 
    for i, data_i in enumerate(train_loader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        original = data_i['original'].cuda()
        semantic = data_i['semantic'].cuda()
        synthesis = data_i['synthesis'].cuda()
        label = data_i['label'].cuda()
        
        # Training
        model_loss, _ = trainer.run_model_one_step(original, synthesis, semantic, label)
        train_loss += model_loss
        
        # Visualizations
        # TODO (Giancarlo): Need to add visualization tool
    
    avg_train_loss = train_loss / len(train_loader)
    print('Training Loss: %f' % (avg_train_loss))
    print('Starting Validation')
    with torch.no_grad():
        val_loss = 0
        for i, data_i in enumerate(val_loader):
            original = data_i['original'].cuda()
            semantic = data_i['semantic'].cuda()
            synthesis = data_i['synthesis'].cuda()
            label = data_i['label'].cuda()
        
            # Training
            loss, predicitions = trainer.run_validation(original, synthesis, semantic, label)
            val_loss += loss

            if i == 2:
                predicted_tensor = torch.ge(predicitions, 0.5)*255
                label_tensor = label*255
                print(data_i['label_path'])
                label_img = Image.fromarray(label_tensor.squeeze().cpu().numpy()).convert('RGB')
                predicted_img = Image.fromarray(predicted_tensor.squeeze().cpu().numpy()).convert('RGB')
                predicted_img.save('/home/giancarlo/Desktop/testing/pred_image%i.png' % epoch)
                label_img.save('/home/giancarlo/Desktop/testing/label_image%i.png' % epoch)
        avg_val_loss = val_loss / len(val_loader)
        print('Validation Loss: %f' % avg_val_loss)
    
    print('saving the latest model (epoch %d, total_steps %d)' %
          (epoch, iter_counter.total_steps_so_far))
    trainer.save(save_fdr, 'latest', exp_name)
    iter_counter.record_current_iter()
    
    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()
    
    if (epoch % config['logger']['save_epoch_freq'] == 0 or epoch == iter_counter.total_epochs):
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save(save_fdr, epoch, exp_name)

print('Training was successfully finished.')
