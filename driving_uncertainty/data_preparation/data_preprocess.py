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

if not opt.no_segmentation:
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
    
    color_mask_fdr = os.path.join(opt.results_dir, 'color-mask')
    overlap_fdr = os.path.join(opt.results_dir, 'overlap')
    semantic_label_fdr = os.path.join(opt.results_dir, 'semantic_labelIds')
    semantic_fdr = os.path.join(opt.results_dir, 'semantic')
    original_fdr = os.path.join(opt.results_dir, 'original')
    
    if not os.path.exists(color_mask_fdr):
        os.makedirs(color_mask_fdr)
        
    if not os.path.exists(overlap_fdr):
        os.makedirs(overlap_fdr)
    
    if not os.path.exists(semantic_fdr):
        os.makedirs(semantic_fdr)
        
    if not os.path.exists(semantic_label_fdr):
        os.makedirs(semantic_label_fdr)
        
    if not os.path.exists(original_fdr):
        os.makedirs(original_fdr)
        
    # creates temporary folder to adapt format to image synthesis
    if not os.path.exists(os.path.join(opt.results_dir, 'temp')):
        os.makedirs(os.path.join(opt.results_dir, 'temp'))
        os.makedirs(os.path.join(opt.results_dir, 'temp', 'gtFine', 'val'))
        os.makedirs(os.path.join(opt.results_dir, 'temp', 'leftImg8bit', 'val'))
    
    # Loop around all figures
    for img_id, img_name in enumerate(images):
        img_dir = os.path.join(data_dir, img_name)
        img = Image.open(img_dir).convert('RGB')
        img.save(os.path.join(original_fdr, img_name))
        img.save(os.path.join(opt.results_dir, 'temp', 'leftImg8bit', 'val', img_name[:-4] + '_leftImg8bit.png'))
        img_tensor = img_transform(img)
    
        # predict
        with torch.no_grad():
            pred = net(img_tensor.unsqueeze(0).cuda())
            print('%04d/%04d: Segmentation Inference done.' % (img_id + 1, len(images)))
    
        pred = pred.cpu().numpy().squeeze()
        pred = np.argmax(pred, axis=0)
    
        color_name = 'color_mask_' + img_name
        overlap_name = 'overlap_' + img_name
        pred_name = 'pred_mask_' + img_name
    
        # save colorized predictions
        colorized = opt.dataset_cls.colorize_mask(pred)
        #colorized.save(os.path.join(color_mask_fdr, color_name))
    
        # save colorized predictions overlapped on original images
        overlap = cv2.addWeighted(np.array(img), 0.5, np.array(colorized.convert('RGB')), 0.5, 0)
        cv2.imwrite(os.path.join(overlap_fdr, overlap_name), overlap[:, :, ::-1])
    
        # save label-based predictions, e.g. for submission purpose
        label_out = np.zeros_like(pred)
        for label_id, train_id in opt.dataset_cls.id_to_trainid.items():
            label_out[np.where(pred == train_id)] = label_id
        cv2.imwrite(os.path.join(semantic_label_fdr, pred_name), label_out)
        cv2.imwrite(os.path.join(semantic_fdr, pred_name), pred)
        cv2.imwrite(os.path.join(opt.results_dir, 'temp', 'gtFine', 'val', pred_name[:-4] + '_instanceIds.png'), label_out)
        cv2.imwrite(os.path.join(opt.results_dir, 'temp', 'gtFine', 'val', pred_name[:-4] + '_labelIds.png'), label_out)
    
    print('Segmentation Results saved.')

print('Starting Image Synthesis Process')

import sys
sys.path.insert(0, './image_synthesis')
import data
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html

world_size = 1
rank = 0

# Corrects where dataset is in necesary format
opt.dataroot = os.path.join(opt.results_dir, 'temp')

opt.world_size = world_size
opt.gpu = 0
opt.mpdist = False

dataloader = data.create_dataloader(opt, world_size, rank)


model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt, rank)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break
    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1], gray=True)

webpage.save()

synthesis_fdr = os.path.join(opt.results_dir, 'synthesis')

# Cleans output folders and deletes temporary files
if not os.path.exists(synthesis_fdr):
    os.makedirs(synthesis_fdr)

source_fdr = os.path.join(opt.results_dir, 'image-synthesis/test_latest/images/synthesized_image')
for image_name in os.listdir(source_fdr):
    shutil.move(os.path.join(source_fdr,image_name), os.path.join(synthesis_fdr,image_name))
    
shutil.rmtree(os.path.join(opt.results_dir, 'image-synthesis'))
shutil.rmtree(os.path.join(opt.results_dir, 'temp'))


