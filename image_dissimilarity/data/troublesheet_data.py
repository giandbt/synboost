import torch.utils.data as data
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from natsort import natsorted
from torchvision import transforms
import torch

import sys
sys.path.append("..")
import data.cityscapes_labels as cityscapes_labels
from data.augmentations import get_transform, get_base_transform

trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid

def troubleshoot_data(dataroot, preprocess_mode, crop_size=512, aspect_ratio=0.5, only_valid=False,
             roi=False, void=False, num_semantic_classes=19, is_train=True):
    original_paths = [os.path.join(dataroot, 'original', image)
                           for image in os.listdir(os.path.join(dataroot, 'original'))]
    semantic_paths = [os.path.join(dataroot, 'semantic', image)
                           for image in os.listdir(os.path.join(dataroot, 'semantic'))]
    synthesis_paths = [os.path.join(dataroot, 'synthesis', image)
                            for image in os.listdir(os.path.join(dataroot, 'synthesis'))]
    if roi:
        label_paths = [os.path.join(dataroot, 'labels_with_ROI', image)
                            for image in os.listdir(os.path.join(dataroot, 'labels_with_ROI'))]
    elif void:
        label_paths = [os.path.join(dataroot, 'labels_with_void', image)
                            for image in os.listdir(os.path.join(dataroot, 'labels_with_void'))]
    else:
        label_paths = [os.path.join(dataroot, 'labels', image)
                            for image in os.listdir(os.path.join(dataroot, 'labels'))]
    # We need to sort the images to ensure all the pairs match with each other
    original_paths = natsorted(original_paths)
    semantic_paths = natsorted(semantic_paths)
    synthesis_paths = natsorted(synthesis_paths)
    label_paths = natsorted(label_paths)

    index = 0

    # get and open all images
    label_path = label_paths[index]
    label = Image.open(label_path)

    semantic_path = semantic_paths[index]
    semantic = Image.open(semantic_path)

    image_path = original_paths[index]
    image = Image.open(image_path).convert('RGB')

    syn_image_path = synthesis_paths[index]
    syn_image = Image.open(syn_image_path).convert('RGB')

    # get input for transformations
    w = crop_size
    h = round(crop_size / aspect_ratio)
    image_size = (h, w)
    if is_train:
        base_transforms = get_base_transform(image_size, 'base_train')
    else:
        base_transforms = get_base_transform(image_size, 'base_test')

    augmentations = get_transform(preprocess_mode)

    import pdb; pdb.set_trace()
    # apply base transformations
    label_tensor = base_transforms(label) * 255
    semantic_tensor = base_transforms(semantic) * 255
    image_tensor = base_transforms(image)
    syn_image_tensor = base_transforms(syn_image)

    # apply augmentations
    if is_train and preprocess_mode != 'none':
        print(preprocess_mode)
        image_tensor = augmentations(transforms.ToPILImage()(image_tensor))
        syn_image_tensor = augmentations(transforms.ToPILImage()(syn_image_tensor))

    # ignore labels classify as void if void is not use for training
    if not void:
        label_tensor[semantic_tensor == 255] = 255

    ## post processing for semantic labels
    if num_semantic_classes == 19:
        semantic_tensor[semantic_tensor == 255] = num_semantic_classes + 1  # 'ignore label is 20'
    semantic_tensor = one_hot_encoding(semantic_tensor, num_semantic_classes + 1)

    input_dict = {'label': label_tensor,
                  'original': image_tensor,
                  'semantic': semantic_tensor,
                  'synthesis': syn_image_tensor,
                  'label_path': label_path,
                  'original_path': image_path,
                  'semantic_path': semantic_path,
                  'syn_image_path': syn_image_path,
                  }

    return input_dict

def one_hot_encoding(semantic, num_classes=20):
    one_hot = torch.zeros(num_classes, semantic.size(1), semantic.size(2))
    for class_id in range(num_classes):
        one_hot[class_id,:,:] = (semantic.squeeze(0)==class_id)
    one_hot = one_hot[:num_classes-1,:,:]
    return one_hot


if __name__ == '__main__':
    from torchvision.transforms import ToPILImage
    import torch
    import sys

    dataset_args = {
        'dataroot': '/home/giancarlo/data/innosuisse/fs_lost_and_found',
        'preprocess_mode': 'none',
        'crop_size': 512,
        'aspect_ratio': 2,
        'void': False,
        'num_semantic_classes': 19,
        'is_train': False
    }
    troubleshoot_data(**dataset_args)
