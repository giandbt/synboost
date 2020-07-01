import torch.utils.data as data
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from natsort import natsorted
from torchvision import transforms
import torch

import sys
sys.path.append("../..")
import image_dissimilarity.data.cityscapes_labels as cityscapes_labels
from image_dissimilarity.data.augmentations import get_transform

trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid

def troubleshoot_data(dataroot, preprocess_mode, crop_size=512, aspect_ratio= 0.5, flip=False, normalize=False,
                 prior = False, only_valid = False, roi = False, light_data= False, void = False, num_semantic_classes = 19, is_train = True):
    original_paths = [os.path.join(dataroot, 'original', image)
                           for image in os.listdir(os.path.join(dataroot, 'original'))]
    if light_data:
        semantic_paths = [os.path.join(dataroot, 'semantic_icnet', image)
                               for image in os.listdir(os.path.join(dataroot, 'semantic_icnet'))]
        synthesis_paths = [os.path.join(dataroot, 'synthesis_spade', image)
                                for image in os.listdir(os.path.join(dataroot, 'synthesis_spade'))]
    else:
        semantic_paths = [os.path.join(dataroot, 'semantic', image)
                               for image in os.listdir(os.path.join(dataroot, 'semantic'))]
        synthesis_paths = [os.path.join(dataroot, 'synthesis', image)
                                for image in os.listdir(os.path.join(dataroot, 'synthesis'))]
    if roi:
        label_paths = [os.path.join(dataroot, 'labels_with_ROI', image)
                            for image in os.listdir(os.path.join(dataroot, 'labels_with_ROI'))]
    elif void:
        label_paths = [os.path.join(dataroot, 'labels_with_void_no_ego', image)
                            for image in os.listdir(os.path.join(dataroot, 'labels_with_void_no_ego'))]
    else:
        label_paths = [os.path.join(dataroot, 'labels', image)
                            for image in os.listdir(os.path.join(dataroot, 'labels'))]
    if prior:
        if light_data:
            mae_features_paths = [os.path.join(dataroot, 'mae_features_spade', image)
                                       for image in os.listdir(os.path.join(dataroot, 'mae_features_spade'))]
            entropy_paths = [os.path.join(dataroot, 'entropy_icnet', image)
                                  for image in os.listdir(os.path.join(dataroot, 'entropy_icnet'))]
            logit_distance_paths = [os.path.join(dataroot, 'logit_distance_icnet', image)
                                         for image in os.listdir(os.path.join(dataroot, 'logit_distance_icnet'))]
        else:
            mae_features_paths = [os.path.join(dataroot, 'mae_features', image)
                                       for image in os.listdir(os.path.join(dataroot, 'mae_features'))]
            entropy_paths = [os.path.join(dataroot, 'entropy', image)
                                  for image in os.listdir(os.path.join(dataroot, 'entropy'))]
            logit_distance_paths = [os.path.join(dataroot, 'logit_distance', image)
                                         for image in os.listdir(os.path.join(dataroot, 'logit_distance'))]
    
    # We need to sort the images to ensure all the pairs match with each other
    original_paths = natsorted(original_paths)
    semantic_paths = natsorted(semantic_paths)
    synthesis_paths = natsorted(synthesis_paths)
    label_paths = natsorted(label_paths)
    if prior:
        mae_features_paths = natsorted(mae_features_paths)
        entropy_paths = natsorted(entropy_paths)
        logit_distance_paths = natsorted(logit_distance_paths)
    

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

    if prior:
        mae_path = mae_features_paths[index]
        mae_image = Image.open(mae_path)
    
        entropy_path = entropy_paths[index]
        entropy_image = Image.open(entropy_path)
    
        distance_path = logit_distance_paths[index]
        distance_image = Image.open(distance_path)
    
    import pdb; pdb.set_trace()
    # get input for transformations
    w = crop_size
    h = round(crop_size / aspect_ratio)
    image_size = (h, w)

    # get augmentations
    base_transforms, augmentations = get_transform(image_size, preprocess_mode)

    # apply base transformations
    label_tensor = base_transforms(label) * 255
    semantic_tensor = base_transforms(semantic) * 255
    syn_image_tensor = base_transforms(syn_image)
    if prior:
        mae_tensor = base_transforms(mae_image)
        entropy_tensor = base_transforms(entropy_image)
        distance_tensor = base_transforms(distance_image)
    else:
        mae_tensor = []
        entropy_tensor = []
        distance_tensor = []

    if is_train and preprocess_mode != 'none':
        image_tensor = augmentations(image)
    else:
        image_tensor = base_transforms(image)

    if normalize:
        norm_transform = transforms.Compose(
            [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # imageNet normamlization
        syn_image_tensor = norm_transform(syn_image_tensor)
        image_tensor = norm_transform(image_tensor)

    # post processing for semantic labels
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
                  'entropy': entropy_tensor,
                  'mae': mae_tensor,
                  'distance': distance_tensor
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
        'dataroot': '/home/giancarlo/data/innosuisse/custom_both',
        'preprocess_mode': 'none',
        'crop_size': 512,
        'aspect_ratio': 2,
        'flip': True,
        'normalize': True,
        'light_data': False,
        'void': False,
        'num_semantic_classes': 19,
        'is_train': False
    }

    troubleshoot_data(**dataset_args)
