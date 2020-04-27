import os
from PIL import Image
import numpy as np
import cv2
from collections import OrderedDict
import shutil
import torch
from torch.backends import cudnn
import torchvision.transforms as transforms

from options.test_options import TestOptions
import sys
sys.path.insert(0, './image_segmentation')
import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

TestOptions = TestOptions()
opt = TestOptions.parse()

assert_and_infer_cfg(opt, train_mode=False)
cudnn.benchmark = False
torch.cuda.empty_cache()

# Get segmentation Net
opt.dataset_cls = cityscapes
net = network.get_net(opt, criterion=None)
net = torch.nn.DataParallel(net).cuda()
print('Segmentation Net built.')
net, _ = restore_snapshot(net, optimizer=None, snapshot=opt.snapshot, restore_optimizer_bool=False)
net.eval()
print('Segmentation Net Restored.')

# Get RGB Original Images
data_dir = opt.demo_folder
images = os.listdir(data_dir)
if len(images) == 0:
    print('There are no images at directory %s. Check the data path.' % (data_dir))
else:
    print('There are %d images to be processed.' % (len(images)))
images.sort()

# Transform images to Tensor based on ImageNet Mean and STD
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])

# Create save directory
if not os.path.exists(opt.results_dir):
    os.makedirs(opt.results_dir)

soft_fdr = os.path.join(opt.results_dir, 'soft_segmentation')

if not os.path.exists(soft_fdr):
    os.makedirs(soft_fdr)


softmax = torch.nn.Softmax(dim=1)

# Loop around all figures
for img_id, img_name in enumerate(images):
    img_dir = os.path.join(data_dir, img_name)
    img = Image.open(img_dir).convert('RGB')
    img_tensor = img_transform(img)

    # predict
    with torch.no_grad():
        pred = net(img_tensor.unsqueeze(0).cuda())
        print('%04d/%04d: Segmentation Inference done.' % (img_id + 1, len(images)))

        outputs = softmax(pred)
        (softmax_pred, predictions) = torch.max(outputs,dim=1)

    pred_og = pred.cpu().numpy().squeeze()
    softmax_pred_og = softmax_pred.cpu().numpy().squeeze()
    predictions_og = predictions.cpu().numpy().squeeze()
    pred = np.argmax(pred_og, axis=0)

    softmax_pred_og = softmax_pred_og* 255
    pred_name = 'soft_prob_' + img_name
    cv2.imwrite(os.path.join(soft_fdr, pred_name), softmax_pred_og)

print('Segmentation Results saved.')




