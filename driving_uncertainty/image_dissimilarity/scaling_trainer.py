import argparse
import yaml
import os

import torch.backends.cudnn as cudnn
import torch

from util import trainer_util, metrics
from util.temperature_scaling import ModelWithTemperature
from models.dissimilarity_model import DissimNet, GuidedDissimNet, ResNetDissimNet, CorrelatedDissimNet

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
opts = parser.parse_args()
cudnn.benchmark = True

# Load experiment setting
with open(opts.config, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

# Activate GPUs
config['gpu_ids'] = opts.gpu_ids
gpu_info = trainer_util.activate_gpus(config)

# Get data loaders
cfg_val_loader = config['val_dataloader']
cfg_test_loader1 = config['test_dataloader1']
cfg_test_loader2 = config['test_dataloader2']
cfg_test_loader3 = config['test_dataloader3']
cfg_test_loader4 = config['test_dataloader4']

val_loader = trainer_util.get_dataloader(cfg_val_loader['dataset_args'], cfg_val_loader['dataloader_args'])
test_loader1 = trainer_util.get_dataloader(cfg_test_loader1['dataset_args'], cfg_test_loader1['dataloader_args'])
test_loader2 = trainer_util.get_dataloader(cfg_test_loader2['dataset_args'], cfg_test_loader2['dataloader_args'])
test_loader3 = trainer_util.get_dataloader(cfg_test_loader3['dataset_args'], cfg_test_loader3['dataloader_args'])
test_loader4 = trainer_util.get_dataloader(cfg_test_loader4['dataset_args'], cfg_test_loader4['dataloader_args'])

# get model
if 'vgg' in config['model']['architecture'] and 'guided' in config['model']['architecture']:
    diss_model = GuidedDissimNet(**config['model']).cuda()
if 'vgg' in config['model']['architecture'] and 'correlated' in config['model']['architecture']:
    diss_model = CorrelatedDissimNet(**config['model']).cuda()
elif 'vgg' in config['model']['architecture']:
    diss_model = DissimNet(**config['model']).cuda()
elif 'resnet' in config['model']['architecture']:
    diss_model = ResNetDissimNet(**config['model']).cuda()
else:
    raise NotImplementedError()

# get pre-trained model
pretrain_config = config['diss_pretrained']
if pretrain_config['load']:
    epoch = pretrain_config['which_epoch']
    save_ckpt_fdr = pretrain_config['save_folder']
    ckpt_name = pretrain_config['experiment_name']

    print('Loading pretrained weights from %s (epoch: %s)' % (ckpt_name, epoch))
    model_path = os.path.join(save_ckpt_fdr, ckpt_name, '%s_net_%s.pth' % (epoch, ckpt_name))
    model_weights = torch.load(model_path)
    diss_model.load_state_dict(model_weights, strict=False)
    # NOTE: For old models, there were some correlation weights created that were not used in the foward pass. That's the reason to include strict=False

orig_model = diss_model # create an uncalibrated model somehow
valid_loader = test_loader1 # Create a DataLoader from the SAME VALIDATION SET used to train orig_model

scaled_model = ModelWithTemperature(orig_model)
scaled_model.set_temperature(valid_loader)

save_path = '/home/giancarlo/code/scaled_model.pth'
torch.save(scaled_model.state_dict(), save_path)
