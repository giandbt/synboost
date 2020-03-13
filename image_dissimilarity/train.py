import argparse
import yaml
import os
from tqdm import tqdm
import numpy as np
import shutil

import torch.backends.cudnn as cudnn
import torch
from torch.utils.tensorboard import SummaryWriter

from trainers.dissimilarity_trainer import DissimilarityTrainer
from util import trainer_util
from util import trainer_util, metrics
from util.iter_counter import IterationCounter

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
opts = parser.parse_args()
cudnn.benchmark = True

# Load experiment setting
with open(opts.config, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
    
# get experiment information
exp_name = config['experiment_name']
save_fdr = config['save_folder']
logs_fdr = config['logger']['results_dir']

print('Starting experiment named: %s'%exp_name)

if not os.path.isdir(save_fdr):
    os.mkdir(save_fdr)

if not os.path.isdir(logs_fdr):
    os.mkdir(logs_fdr)
    
train_writer = SummaryWriter(os.path.join(logs_fdr, exp_name, 'train'), flush_secs=30)
val_writer = SummaryWriter(os.path.join(logs_fdr, exp_name, 'validation'), flush_secs=30)
test_writer = SummaryWriter(os.path.join(logs_fdr, exp_name, 'test'), flush_secs=30)

# Save config file use for experiment
shutil.copy(opts.config, os.path.join(logs_fdr, exp_name, 'config.yaml'))

# Activate GPUs
config['gpu_ids'] = opts.gpu_ids
gpu_info = trainer_util.activate_gpus(config)

# Get data loaders
cfg_train_loader = config['train_dataloader']
cfg_val_loader = config['val_dataloader']
cfg_test_loader1 = config['test_dataloader1']
cfg_test_loader2 = config['test_dataloader2']
cfg_test_loader3 = config['test_dataloader3']
train_loader = trainer_util.get_dataloader(cfg_train_loader['dataset_args'], cfg_train_loader['dataloader_args'])
val_loader = trainer_util.get_dataloader(cfg_val_loader['dataset_args'], cfg_val_loader['dataloader_args'])
test_loader1 = trainer_util.get_dataloader(cfg_test_loader1['dataset_args'], cfg_test_loader1['dataloader_args'])
test_loader2 = trainer_util.get_dataloader(cfg_test_loader2['dataset_args'], cfg_test_loader2['dataloader_args'])
test_loader3 = trainer_util.get_dataloader(cfg_test_loader3['dataset_args'], cfg_test_loader3['dataloader_args'])

# Getting parameters for test
dataset = cfg_test_loader1['dataset_args']
h = int((dataset['crop_size']/dataset['aspect_ratio']))
w = int(dataset['crop_size'])

# create trainer for our model
print('Loading Model')
trainer = DissimilarityTrainer(config)

# create tool for counting iterations
batch_size = config['train_dataloader']['dataloader_args']['batch_size']
iter_counter = IterationCounter(config, len(train_loader), batch_size)

print('Starting Training...')
best_val_loss = float('inf')
iter = 0
for epoch in iter_counter.training_epochs():
    print('Starting Epoch #%i for experiment %s'% (epoch, exp_name))
    iter_counter.record_epoch_start(epoch)
    train_loss = 0 
    for i, data_i in enumerate(tqdm(train_loader), start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        original = data_i['original'].cuda()
        semantic = data_i['semantic'].cuda()
        synthesis = data_i['synthesis'].cuda()
        label = data_i['label'].cuda()
        
        # Training
        model_loss, _ = trainer.run_model_one_step(original, synthesis, semantic, label)
        train_loss += model_loss
        train_writer.add_scalar('Loss_iter', model_loss, iter)
        iter+=1
        
    avg_train_loss = train_loss / len(train_loader)
    train_writer.add_scalar('Loss_epoch', avg_train_loss, epoch)
    
    print('Training Loss: %f' % (avg_train_loss))
    print('Starting Validation')
    with torch.no_grad():
        val_loss = 0
        for i, data_i in enumerate(tqdm(val_loader)):
            original = data_i['original'].cuda()
            semantic = data_i['semantic'].cuda()
            synthesis = data_i['synthesis'].cuda()
            label = data_i['label'].cuda()
        
            # Evaluating
            loss, _ = trainer.run_validation(original, synthesis, semantic, label)
            val_loss += loss
            
        avg_val_loss = val_loss / len(val_loader)
        print('Validation Loss: %f' % avg_val_loss)

        val_writer.add_scalar('Loss_epoch', avg_val_loss, epoch)
        
        if avg_val_loss < best_val_loss:
            print('Validation loss for epoch %d (%f) is better than previous best loss (%f). Saving best model.'
                  %(epoch, avg_val_loss, best_val_loss))
            best_val_loss = avg_val_loss
            trainer.save(save_fdr, 'best', exp_name)
    
    # Starts Testing (Test Set 1)
        print('Starting Testing For %s' % os.path.basename(cfg_test_loader1['dataset_args']['dataroot']))
        flat_pred = np.zeros(w * h * len(test_loader1))
        flat_labels = np.zeros(w * h * len(test_loader1))
        for i, data_i in enumerate(tqdm(test_loader1)):
            original = data_i['original'].cuda()
            semantic = data_i['semantic'].cuda()
            synthesis = data_i['synthesis'].cuda()
            label = data_i['label'].cuda()
        
            # Evaluating
            loss, outputs = trainer.run_validation(original, synthesis, semantic, label)
            (softmax_pred, predictions) = torch.max(outputs, dim=1)
            flat_pred[i * w * h:i * w * h + w * h] = torch.flatten(outputs[:, 1, :, :]).detach().cpu().numpy()
            flat_labels[i * w * h:i * w * h + w * h] = torch.flatten(label).detach().cpu().numpy()

        if config['test_dataloader1']['dataset_args']['roi']:
            invalid_indices = np.argwhere(flat_labels == 255)
            flat_labels = np.delete(flat_labels, invalid_indices)
            flat_pred = np.delete(flat_pred, invalid_indices)
            
        print('Calculating metrics')
        results = metrics.get_metrics(flat_labels, flat_pred)
        print('AU_ROC: %f' % results['auroc'])
        print('mAP: %f' % results['AP'])
        print('FPR@95TPR: %f' % results['FPR@95%TPR'])
    
        test_writer.add_scalar('%s AUC_ROC' % os.path.basename(cfg_test_loader1['dataset_args']['dataroot']), results['auroc'], epoch)
        test_writer.add_scalar('%s mAP' % os.path.basename(cfg_test_loader1['dataset_args']['dataroot']), results['AP'], epoch)
        test_writer.add_scalar('%s FPR@95TPR' % os.path.basename(cfg_test_loader1['dataset_args']['dataroot']), results['FPR@95%TPR'], epoch)

        # Starts Testing (Test Set 2)
        print('Starting Testing For %s' % os.path.basename(cfg_test_loader2['dataset_args']['dataroot']))
        flat_pred = np.zeros(w * h * len(test_loader2))
        flat_labels = np.zeros(w * h * len(test_loader2))
        for i, data_i in enumerate(tqdm(test_loader2)):
            original = data_i['original'].cuda()
            semantic = data_i['semantic'].cuda()
            synthesis = data_i['synthesis'].cuda()
            label = data_i['label'].cuda()
    
            # Evaluating
            loss, outputs = trainer.run_validation(original, synthesis, semantic, label)
            (softmax_pred, predictions) = torch.max(outputs, dim=1)
            flat_pred[i * w * h:i * w * h + w * h] = torch.flatten(outputs[:, 1, :, :]).detach().cpu().numpy()
            flat_labels[i * w * h:i * w * h + w * h] = torch.flatten(label).detach().cpu().numpy()

        if config['test_dataloader2']['dataset_args']['roi']:
            invalid_indices = np.argwhere(flat_labels == 255)
            flat_labels = np.delete(flat_labels, invalid_indices)
            flat_pred = np.delete(flat_pred, invalid_indices)

        print('Calculating metrics')
        results = metrics.get_metrics(flat_labels, flat_pred)
        print('AU_ROC: %f' % results['auroc'])
        print('mAP: %f' % results['AP'])
        print('FPR@95TPR: %f' % results['FPR@95%TPR'])

        test_writer.add_scalar('%s AUC_ROC' % os.path.basename(cfg_test_loader2['dataset_args']['dataroot']),
                               results['auroc'], epoch)
        test_writer.add_scalar('%s mAP' % os.path.basename(cfg_test_loader2['dataset_args']['dataroot']), results['AP'],
                               epoch)
        test_writer.add_scalar('%s FPR@95TPR' % os.path.basename(cfg_test_loader2['dataset_args']['dataroot']),
                               results['FPR@95%TPR'], epoch)
        
        # Starts Testing (Test Set 3)
        print('Starting Testing For %s' % os.path.basename(cfg_test_loader3['dataset_args']['dataroot']))
        flat_pred = np.zeros(w * h * len(test_loader3))
        flat_labels = np.zeros(w * h * len(test_loader3))
        for i, data_i in enumerate(tqdm(test_loader3)):
            original = data_i['original'].cuda()
            semantic = data_i['semantic'].cuda()
            synthesis = data_i['synthesis'].cuda()
            label = data_i['label'].cuda()
    
            # Evaluating
            loss, outputs = trainer.run_validation(original, synthesis, semantic, label)
            (softmax_pred, predictions) = torch.max(outputs, dim=1)
            flat_pred[i * w * h:i * w * h + w * h] = torch.flatten(outputs[:, 1, :, :]).detach().cpu().numpy()
            flat_labels[i * w * h:i * w * h + w * h] = torch.flatten(label).detach().cpu().numpy()

        if config['test_dataloader3']['dataset_args']['roi']:
            invalid_indices = np.argwhere(flat_labels == 255)
            flat_labels = np.delete(flat_labels, invalid_indices)
            flat_pred = np.delete(flat_pred, invalid_indices)

        print('Calculating metrics')
        results = metrics.get_metrics(flat_labels, flat_pred)
        print('AU_ROC: %f' % results['auroc'])
        print('mAP: %f' % results['AP'])
        print('FPR@95TPR: %f' % results['FPR@95%TPR'])

        test_writer.add_scalar('%s AUC_ROC' % os.path.basename(cfg_test_loader3['dataset_args']['dataroot']),
                               results['auroc'], epoch)
        test_writer.add_scalar('%s mAP' % os.path.basename(cfg_test_loader3['dataset_args']['dataroot']), results['AP'],
                               epoch)
        test_writer.add_scalar('%s FPR@95TPR' % os.path.basename(cfg_test_loader3['dataset_args']['dataroot']),
                               results['FPR@95%TPR'], epoch)
    
    print('saving the latest model (epoch %d, total_steps %d)' %
          (epoch, iter_counter.total_steps_so_far))
    trainer.save(save_fdr, 'latest', exp_name)
    
    #trainer.update_learning_rate(epoch) TODO (Giancarlo): Have a more permanent solution, currently we have ReduceonPlateu and Manually modification of LR
    trainer.update_learning_rate_schedule(avg_val_loss)
    iter_counter.record_epoch_end()
    
    if (epoch % config['logger']['save_epoch_freq'] == 0 or epoch == iter_counter.total_epochs):
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save(save_fdr, epoch, exp_name)
        
train_writer.close()
val_writer.close()
print('Training was successfully finished.')
