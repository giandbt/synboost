import argparse
import yaml
import os
from tqdm import tqdm
import numpy as np
import shutil
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage, ToTensor

from trainers.dissimilarity_trainer import DissimilarityTrainer
from util import trainer_util
from util import trainer_util, metrics
from util.iter_counter import IterationCounter
from util.image_logging import ImgLogging
from util import visualization

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--seed', type=str, default='0', help='seed for experiment')
opts = parser.parse_args()
cudnn.benchmark = True

# Load experiment setting
with open(opts.config, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
    
# get experiment information
exp_name = config['experiment_name'] + opts.seed
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
cfg_test_loader4 = config['test_dataloader4']

# checks if we are using prior images
prior = config['model']['prior']

# adds logic to dataloaders (avoid repetition in config file)
cfg_train_loader['dataset_args']['prior'] = prior
cfg_val_loader['dataset_args']['prior'] = prior
cfg_test_loader1['dataset_args']['prior'] = prior
cfg_test_loader2['dataset_args']['prior'] = prior
cfg_test_loader3['dataset_args']['prior'] = prior
cfg_test_loader4['dataset_args']['prior'] = prior
    
    
train_loader = trainer_util.get_dataloader(cfg_train_loader['dataset_args'], cfg_train_loader['dataloader_args'])
val_loader = trainer_util.get_dataloader(cfg_val_loader['dataset_args'], cfg_val_loader['dataloader_args'])
test_loader1 = trainer_util.get_dataloader(cfg_test_loader1['dataset_args'], cfg_test_loader1['dataloader_args'])
test_loader2 = trainer_util.get_dataloader(cfg_test_loader2['dataset_args'], cfg_test_loader2['dataloader_args'])
test_loader3 = trainer_util.get_dataloader(cfg_test_loader3['dataset_args'], cfg_test_loader3['dataloader_args'])
test_loader4 = trainer_util.get_dataloader(cfg_test_loader4['dataset_args'], cfg_test_loader4['dataloader_args'])

if config['training_strategy']['image_visualization']:
    cfg_image_loader = config['img_dataloader']
    image_writer = SummaryWriter(os.path.join(logs_fdr, exp_name, 'images'), flush_secs=30)
    image_loader = trainer_util.get_dataloader(cfg_image_loader['dataset_args'], cfg_image_loader['dataloader_args'])
    image_logger = ImgLogging(cfg_image_loader['dataset_args']['preprocess_mode'])

# Getting parameters for test
dataset = cfg_test_loader1['dataset_args']
h = int((dataset['crop_size']/dataset['aspect_ratio']))
w = int(dataset['crop_size'])

# create trainer for our model
print('Loading Model')
trainer = DissimilarityTrainer(config, seed=int(opts.seed))

# create tool for counting iterations
batch_size = config['train_dataloader']['dataloader_args']['batch_size']
iter_counter = IterationCounter(config, len(train_loader), batch_size)

# Softmax layer for testing
softmax = torch.nn.Softmax(dim=1)

print('Starting Training...')
best_val_loss = float('inf')
best_map_metric = 0
iter = 0
for epoch in iter_counter.training_epochs():
    
    print('Starting Epoch #%i for experiment %s'% (epoch, exp_name))
    iter_counter.record_epoch_start(epoch)
    train_loss = 0
    cumul_map_sum = 0
    for i, data_i in enumerate(tqdm(train_loader), start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        original = data_i['original'].cuda()
        semantic = data_i['semantic'].cuda()
        synthesis = data_i['synthesis'].cuda()
        label = data_i['label'].cuda()
        
        # Training
        if prior:
            entropy = data_i['entropy'].cuda()
            mae = data_i['mae'].cuda()
            distance = data_i['distance'].cuda()
            model_loss, _ = trainer.run_model_one_step_prior(original, synthesis, semantic, label, entropy, mae, distance)
        else:
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
            
            if prior:
                entropy = data_i['entropy'].cuda()
                mae = data_i['mae'].cuda()
                distance = data_i['distance'].cuda()
    
                # Evaluating
                loss, _ = trainer.run_validation_prior(original, synthesis, semantic, label, entropy, mae,
                                                             distance)
            else:
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
        val_loss = 0
        for i, data_i in enumerate(tqdm(test_loader1)):
            original = data_i['original'].cuda()
            semantic = data_i['semantic'].cuda()
            synthesis = data_i['synthesis'].cuda()
            label = data_i['label'].cuda()
            
            if prior:
                entropy = data_i['entropy'].cuda()
                mae = data_i['mae'].cuda()
                distance = data_i['distance'].cuda()
    
                # Evaluating
                loss, outputs = trainer.run_validation_prior(original, synthesis, semantic, label, entropy, mae,
                                                             distance)
            else:
                loss, outputs = trainer.run_validation(original, synthesis, semantic, label)
                
            val_loss += loss
            outputs = softmax(outputs)
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

        avg_val_loss = val_loss / len(test_loader1)
        test_writer.add_scalar('%s AUC_ROC' % os.path.basename(cfg_test_loader1['dataset_args']['dataroot']), results['auroc'], epoch)
        test_writer.add_scalar('%s mAP' % os.path.basename(cfg_test_loader1['dataset_args']['dataroot']), results['AP'], epoch)
        test_writer.add_scalar('%s FPR@95TPR' % os.path.basename(cfg_test_loader1['dataset_args']['dataroot']), results['FPR@95%TPR'], epoch)
        test_writer.add_scalar('val_loss_%s' % os.path.basename(cfg_test_loader1['dataset_args']['dataroot']), avg_val_loss, epoch)
        cumul_map_sum += results['AP']
        # Starts Testing (Test Set 2)
        print('Starting Testing For %s' % os.path.basename(cfg_test_loader2['dataset_args']['dataroot']))
        flat_pred = np.zeros(w * h * len(test_loader2))
        flat_labels = np.zeros(w * h * len(test_loader2))
        val_loss = 0
        for i, data_i in enumerate(tqdm(test_loader2)):
            original = data_i['original'].cuda()
            semantic = data_i['semantic'].cuda()
            synthesis = data_i['synthesis'].cuda()
            label = data_i['label'].cuda()
            
            if prior:
                entropy = data_i['entropy'].cuda()
                mae = data_i['mae'].cuda()
                distance = data_i['distance'].cuda()
    
                # Evaluating
                loss, outputs = trainer.run_validation_prior(original, synthesis, semantic, label, entropy, mae,
                                                             distance)
            else:
                loss, outputs = trainer.run_validation(original, synthesis, semantic, label)
                
            val_loss += loss
            outputs = softmax(outputs)
            (softmax_pred, predictions) = torch.max(outputs, dim=1)
            flat_pred[i * w * h:i * w * h + w * h] = torch.flatten(outputs[:, 1, :, :]).detach().cpu().numpy()
            flat_labels[i * w * h:i * w * h + w * h] = torch.flatten(label).detach().cpu().numpy()

        if config['test_dataloader2']['dataset_args']['roi']:
            invalid_indices = np.argwhere(flat_labels == 255)
            flat_labels = np.delete(flat_labels, invalid_indices)
            flat_pred = np.delete(flat_pred, invalid_indices)

        avg_val_loss = val_loss / len(test_loader2)

        print('Calculating metrics')
        results = metrics.get_metrics(flat_labels, flat_pred)
        print('AU_ROC: %f' % results['auroc'])
        print('mAP: %f' % results['AP'])
        print('FPR@95TPR: %f' % results['FPR@95%TPR'])

        cumul_map_sum += results['AP']
        test_writer.add_scalar('%s AUC_ROC' % os.path.basename(cfg_test_loader2['dataset_args']['dataroot']),
                               results['auroc'], epoch)
        test_writer.add_scalar('%s mAP' % os.path.basename(cfg_test_loader2['dataset_args']['dataroot']), results['AP'],
                               epoch)
        test_writer.add_scalar('%s FPR@95TPR' % os.path.basename(cfg_test_loader2['dataset_args']['dataroot']),
                               results['FPR@95%TPR'], epoch)
        test_writer.add_scalar('val_loss_%s' % os.path.basename(cfg_test_loader2['dataset_args']['dataroot']),
                               avg_val_loss, epoch)

        # Starts Testing (Test Set 3)
        print('Starting Testing For %s' % os.path.basename(cfg_test_loader3['dataset_args']['dataroot']))
        flat_pred = np.zeros(w * h * len(test_loader3))
        flat_labels = np.zeros(w * h * len(test_loader3))
        val_loss = 0
        for i, data_i in enumerate(tqdm(test_loader3)):
            original = data_i['original'].cuda()
            semantic = data_i['semantic'].cuda()
            synthesis = data_i['synthesis'].cuda()
            label = data_i['label'].cuda()
            
            if prior:
                entropy = data_i['entropy'].cuda()
                mae = data_i['mae'].cuda()
                distance = data_i['distance'].cuda()
    
                # Evaluating
                loss, outputs = trainer.run_validation_prior(original, synthesis, semantic, label, entropy, mae,
                                                             distance)
            else:
                loss, outputs = trainer.run_validation(original, synthesis, semantic, label)
                
            val_loss += loss
            outputs = softmax(outputs)
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
        cumul_map_sum += results['AP']
        avg_val_loss = val_loss / len(test_loader3)

        test_writer.add_scalar('%s AUC_ROC' % os.path.basename(cfg_test_loader3['dataset_args']['dataroot']),
                               results['auroc'], epoch)
        test_writer.add_scalar('%s mAP' % os.path.basename(cfg_test_loader3['dataset_args']['dataroot']), results['AP'],
                               epoch)
        test_writer.add_scalar('%s FPR@95TPR' % os.path.basename(cfg_test_loader3['dataset_args']['dataroot']),
                               results['FPR@95%TPR'], epoch)
        test_writer.add_scalar('val_loss_%s' % os.path.basename(cfg_test_loader3['dataset_args']['dataroot']),
                               avg_val_loss, epoch)

        if cumul_map_sum > best_map_metric:
            print('Cumulative mAP for epoch %d (%f) is better than previous best mAP (%f). Saving best model.'
                  % (epoch, cumul_map_sum, best_map_metric))
            best_map_metric = cumul_map_sum
            trainer.save(save_fdr, 'best_map', exp_name)

        # Starts Testing (Test Set 4)
        print('Starting Testing For %s' % os.path.basename(cfg_test_loader4['dataset_args']['dataroot']))
        flat_pred = np.zeros(w * h * len(test_loader4))
        flat_labels = np.zeros(w * h * len(test_loader4))
        val_loss = 0
        for i, data_i in enumerate(tqdm(test_loader4)):
            original = data_i['original'].cuda()
            semantic = data_i['semantic'].cuda()
            synthesis = data_i['synthesis'].cuda()
            label = data_i['label'].cuda()
            
            if prior:
                entropy = data_i['entropy'].cuda()
                mae = data_i['mae'].cuda()
                distance = data_i['distance'].cuda()

                # Evaluating
                loss, outputs = trainer.run_validation_prior(original, synthesis, semantic, label, entropy, mae, distance)
            else:
                loss, outputs = trainer.run_validation(original, synthesis, semantic, label)
                
            val_loss += loss
            outputs = softmax(outputs)
            (softmax_pred, predictions) = torch.max(outputs, dim=1)
            flat_pred[i * w * h:i * w * h + w * h] = torch.flatten(outputs[:, 1, :, :]).detach().cpu().numpy()
            flat_labels[i * w * h:i * w * h + w * h] = torch.flatten(label).detach().cpu().numpy()

        if config['test_dataloader4']['dataset_args']['roi']:
            invalid_indices = np.argwhere(flat_labels == 255)
            flat_labels = np.delete(flat_labels, invalid_indices)
            flat_pred = np.delete(flat_pred, invalid_indices)

        print('Calculating metrics')
        results = metrics.get_metrics(flat_labels, flat_pred)
        print('AU_ROC: %f' % results['auroc'])
        print('mAP: %f' % results['AP'])
        print('FPR@95TPR: %f' % results['FPR@95%TPR'])

        avg_val_loss = val_loss / len(test_loader4)

        test_writer.add_scalar('%s AUC_ROC' % os.path.basename(cfg_test_loader4['dataset_args']['dataroot']),
                               results['auroc'], epoch)
        test_writer.add_scalar('%s mAP' % os.path.basename(cfg_test_loader4['dataset_args']['dataroot']), results['AP'],
                               epoch)
        test_writer.add_scalar('%s FPR@95TPR' % os.path.basename(cfg_test_loader4['dataset_args']['dataroot']),
                               results['FPR@95%TPR'], epoch)
        test_writer.add_scalar('val_loss_%s' % os.path.basename(cfg_test_loader4['dataset_args']['dataroot']),
                               avg_val_loss, epoch)

        # Starts Image Visualization Module
        if config['training_strategy']['image_visualization']:
            print('Starting Visualization For %s' % os.path.basename(cfg_image_loader['dataset_args']['dataroot']))
            for i, data_i in enumerate(tqdm(image_loader)):
                original = data_i['original'].cuda()
                semantic = data_i['semantic'].cuda()
                synthesis = data_i['synthesis'].cuda()
                label = data_i['label'].cuda()
                entropy = data_i['entropy'].cuda()
                mae = data_i['mae'].cuda()
                distance = data_i['distance'].cuda()

                # Evaluating
                if prior:
                    loss, _ = trainer.run_validation(original, synthesis, semantic, label)
                else:
                    loss, _ = trainer.run_validation(original, synthesis, semantic, label, entropy, mae, distance)
                val_loss += loss
                outputs = softmax(outputs)
                (softmax_pred, predictions) = torch.max(outputs, dim=1)

                # post processing for semantic, label and prediction
                semantic_post = torch.zeros([original.shape[0], 3, 256, 512])
                for idx, semantic_ in enumerate(semantic):
                    (_, semantic_) = torch.max(semantic_, dim = 0)
                    semantic_ = 256 - np.asarray(ToPILImage()(semantic_.type(torch.FloatTensor).cpu()))
                    semantic_[semantic_ == 256] = 0
                    semantic_ = visualization.colorize_mask(semantic_)
                    semantic_ = ToTensor()(semantic_.convert('RGB'))
                    semantic_post[idx, :, :, :] = semantic_

                label_post = torch.zeros([original.shape[0], 3, 256, 512])
                for idx, label_ in enumerate(label):
                    label_ = 256 - np.asarray(ToPILImage()(label_.type(torch.FloatTensor).cpu()))
                    # There must be a better way...
                    label_[label_ == 256] = 0
                    label_[label_ == 255] = 100
                    label_[label_ == 1] = 255
                    label_ = ToTensor()(Image.fromarray(label_).convert('RGB'))
                    label_post[idx, :, :, :] = label_

                predictions_post = torch.zeros([original.shape[0], 3, 256, 512])
                for idx, predictions_ in enumerate(predictions):
                    predictions_ = np.asarray(ToPILImage()(predictions_.type(torch.FloatTensor).cpu()))
                    predictions_ = ToTensor()(Image.fromarray(predictions_).convert('RGB'))
                    predictions_post[idx, :, :, :] = predictions_

                all_images = torch.zeros([original.shape[0]*5, 3, 256, 512])
                for idx, (original_img, semantic_img, synthesis_img, label_img, predictions_img) in \
                        enumerate(zip(original, semantic_post, synthesis, label_post, predictions_post)):
                    all_images[idx*5, :, :, :] = original_img
                    all_images[idx*5+1, :, :, :] = semantic_img
                    all_images[idx*5+2, :, :, :] = synthesis_img
                    all_images[idx*5+3, :, :, :] = label_img
                    all_images[idx*5+4, :, :, :] = predictions_img
                grid = make_grid(all_images, 5)
            image_writer.add_image('results', grid, epoch)

    print('saving the latest model (epoch %d, total_steps %d)' %
          (epoch, iter_counter.total_steps_so_far))
    trainer.save(save_fdr, 'latest', exp_name)

    trainer.update_learning_rate_schedule(avg_val_loss)
    iter_counter.record_epoch_end()
    
    if (epoch % config['logger']['save_epoch_freq'] == 0 or epoch == iter_counter.total_epochs):
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save(save_fdr, epoch, exp_name)
        
train_writer.close()
val_writer.close()
test_writer.close()
if config['training_strategy']['image_visualization']:
    image_writer.close()
print('Training was successfully finished.')
