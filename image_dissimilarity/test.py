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

from util import trainer_util, metrics
from util.iter_counter import IterationCounter
from models.dissimilarity_model import DissimNet, DissimNetPrior

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
parser.add_argument('--gpu_ids',
                    type=str,
                    default='0',
                    help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
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
store_fdr_exp = os.path.join(config['store_results'], exp_name)
ensemble = config['ensemble']

if not os.path.isdir(store_fdr):
    os.makedirs(store_fdr, exist_ok=True)

if not os.path.isdir(store_fdr_exp):
    os.makedirs(
        store_fdr_exp,
        exist_ok=True,
    )

if not os.path.isdir(os.path.join(store_fdr_exp, 'pred')):
    os.makedirs(os.path.join(store_fdr_exp, 'label'), exist_ok=True)
    os.makedirs(os.path.join(store_fdr_exp, 'pred'), exist_ok=True)
    os.makedirs(os.path.join(store_fdr_exp, 'soft'), exist_ok=True)

# Activate GPUs
config['gpu_ids'] = opts.gpu_ids
gpu_info = trainer_util.activate_gpus(config)

# checks if we are using prior images
prior = config['model']['prior']
# Get data loaders
cfg_test_loader = config['test_dataloader']
# adds logic to dataloaders (avoid repetition in config file)
cfg_test_loader['dataset_args']['prior'] = prior
test_loader = trainer_util.get_dataloader(cfg_test_loader['dataset_args'],
                                          cfg_test_loader['dataloader_args'])

# get model
if config['model']['prior']:
    diss_model = DissimNetPrior(**config['model']).cuda()
elif 'vgg' in config['model']['architecture']:
    diss_model = DissimNet(**config['model']).cuda()
else:
    raise NotImplementedError()

diss_model.eval()
model_path = os.path.join(save_fdr, exp_name,
                          '%s_net_%s.pth' % (epoch, exp_name))
model_weights = torch.load(model_path)
diss_model.load_state_dict(model_weights)

softmax = torch.nn.Softmax(dim=1)

# create memory locations for results to save time while running the code
dataset = cfg_test_loader['dataset_args']
h = int((dataset['crop_size'] / dataset['aspect_ratio']))
w = int(dataset['crop_size'])
flat_pred = np.zeros(w * h * len(test_loader), dtype='float32')
flat_labels = np.zeros(w * h * len(test_loader), dtype='float32')
num_points = 50

with torch.no_grad():
    for i, data_i in enumerate(tqdm(test_loader)):
        original = data_i['original'].cuda()
        semantic = data_i['semantic'].cuda()
        synthesis = data_i['synthesis'].cuda()
        label = data_i['label'].cuda()

        if prior:
            entropy = data_i['entropy'].cuda()
            mae = data_i['mae'].cuda()
            distance = data_i['distance'].cuda()
            outputs = softmax(
                diss_model(original, synthesis, semantic, entropy, mae,
                           distance))
        else:
            outputs = softmax(diss_model(original, synthesis, semantic))
        (softmax_pred, predictions) = torch.max(outputs, dim=1)
        if ensemble:
            soft_pred = outputs[:, 1, :, :] * 0.75 + entropy * 0.25
        else:
            soft_pred = outputs[:, 1, :, :]
        flat_pred[i * w * h:i * w * h +
                  w * h] = torch.flatten(soft_pred).detach().cpu().numpy()
        flat_labels[i * w * h:i * w * h +
                    w * h] = torch.flatten(label).detach().cpu().numpy()
        # Save results
        predicted_tensor = predictions * 1
        label_tensor = label * 1

        file_name = os.path.basename(data_i['original_path'][0])
        label_img = Image.fromarray(
            label_tensor.squeeze().cpu().numpy().astype(np.uint8))
        soft_img = Image.fromarray(
            (soft_pred.squeeze().cpu().numpy() * 255).astype(np.uint8))
        predicted_img = Image.fromarray(
            predicted_tensor.squeeze().cpu().numpy().astype(np.uint8))
        predicted_img.save(os.path.join(store_fdr_exp, 'pred', file_name))
        soft_img.save(os.path.join(store_fdr_exp, 'soft', file_name))
        label_img.save(os.path.join(store_fdr_exp, 'label', file_name))

print('Calculating metric scores')
if config['test_dataloader']['dataset_args']['roi']:
    invalid_indices = np.argwhere(flat_labels == 255)
    flat_labels = np.delete(flat_labels, invalid_indices)
    flat_pred = np.delete(flat_pred, invalid_indices)

results = metrics.get_metrics(flat_labels, flat_pred)

print("roc_auc_score : " + str(results['auroc']))
print("mAP: " + str(results['AP']))
print("FPR@95%TPR : " + str(results['FPR@95%TPR']))

if config['visualize']:
    plt.figure()
    lw = 2
    plt.plot(results['fpr'],
             results['tpr'],
             color='darkorange',
             lw=lw,
             label='ROC curve (area = %0.2f)' % results['auroc'])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(store_fdr_exp, 'roc_curve.png'))
