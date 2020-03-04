import argparse
import yaml
import torch.backends.cudnn as cudnn
import torch
from PIL import Image
import numpy as np
import os
from sklearn import metrics
import matplotlib.pyplot as plt
from tqdm import tqdm

from util import trainer_util
from util.iter_counter import IterationCounter
from models.dissimilarity_model import DissimNet

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
epoch = config['which_epoch']
store_fdr = config['store_results']

if not os.path.isdir(store_fdr):
    os.mkdir(store_fdr)

if not os.path.isdir(os.path.join(store_fdr, 'pred')):
    os.mkdir(os.path.join(store_fdr, 'label'))
    os.mkdir(os.path.join(store_fdr, 'pred'))

# Activate GPUs
config['gpu_ids'] = opts.gpu_ids
gpu_info = trainer_util.activate_gpus(config)

# Get data loaders
cfg_test_loader = config['test_dataloader']
test_loader = trainer_util.get_dataloader(cfg_test_loader['dataset_args'], cfg_test_loader['dataloader_args'])

# get model
diss_model = DissimNet(**config['model']['parameters']).cuda()
diss_model.eval()
model_path = os.path.join(save_fdr, exp_name, '%s_net_%s.pth' %(epoch, exp_name))
model_weights = torch.load(model_path)
diss_model.load_state_dict(model_weights)

softmax = torch.nn.Softmax(dim=1)

# create memory locations for results to save time while running the code
dataset = cfg_test_loader['dataset_args']
h = int((dataset['crop_size']/dataset['aspect_ratio']))
w = int(dataset['crop_size'])
flat_pred = np.zeros(w*h*len(test_loader), dtype='float32')
flat_labels = np.zeros(w*h*len(test_loader), dtype='float32')

with torch.no_grad():
    for i, data_i in enumerate(tqdm(test_loader)):
        original = data_i['original'].cuda()
        semantic = data_i['semantic'].cuda()
        synthesis = data_i['synthesis'].cuda()
        label = data_i['label'].cuda()
        
        outputs = softmax(diss_model(original, synthesis, semantic))
        (softmax_pred, predictions) = torch.max(outputs,dim=1)
        flat_pred[i*w*h:i*w*h+w*h] = torch.flatten(outputs[:,1,:,:]).detach().cpu().numpy()
        flat_labels[i*w*h:i*w*h+w*h] = torch.flatten(label).detach().cpu().numpy()
    
        # Save results
        predicted_tensor = predictions * 255
        label_tensor = label * 255
        
        file_name = os.path.basename(data_i['original_path'][0])
        label_img = Image.fromarray(label_tensor.squeeze().cpu().numpy().astype(np.uint8)).convert('RGB')
        predicted_img = Image.fromarray(predicted_tensor.squeeze().cpu().numpy().astype(np.uint8)).convert('RGB')
        predicted_img.save(os.path.join(store_fdr, 'pred', file_name))
        label_img.save(os.path.join(store_fdr, 'label', file_name))

print('Calculating AUC-ROC score')
if config['test_dataloader']['dataset_args']['roi']:
    invalid_indices = np.argwhere(flat_labels == 255)
    flat_labels = np.delete(flat_labels, invalid_indices)
    flat_pred = np.delete(flat_pred, invalid_indices)

fpr, tpr, _ = metrics.roc_curve(flat_labels, flat_pred)
roc_auc = metrics.auc(fpr, tpr)
print("roc_auc_score : " + str(roc_auc))

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(store_fdr,'roc_curve.png'))
