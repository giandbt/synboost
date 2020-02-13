import argparse
import yaml
import torch.backends.cudnn as cudnn

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
    # set epoch for data sampler
    train_loader.sampler.set_epoch(epoch)
    
    iter_counter.record_epoch_start(epoch)
    
    for i, data_i in enumerate(train_loader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        
        # Training
        trainer.run_model_one_step(data_i)
        
        # Visualizations
        # TODO (Giancarlo): Need to add visualization tool
    
    print('saving the latest model (epoch %d, total_steps %d)' %
          (epoch, iter_counter.total_steps_so_far))
    trainer.save('latest')
    iter_counter.record_current_iter()
    
    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()
    
    if (epoch % config['logger']['save_epoch_freq'] == 0 or epoch == iter_counter.total_epochs):
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save(epoch)

print('Training was successfully finished.')