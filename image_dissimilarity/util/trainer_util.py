import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2


import sys
sys.path.append("..")
from data.cityscapes_dataset import CityscapesDataset

def activate_gpus(config):
    """Identify which GPUs to activate
        Args:
            config: Configuration dictionary with project hyperparameters
        Returns:
            dict: Required information for GPU/CPU training
    """
    str_ids = config['gpu_ids'].split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    if torch.cuda.is_available() and len(gpu_ids) > 0:
        use_gpu = True
    else:
        use_gpu = False
        gpu_ids = []
    device = torch.device("cuda:" + str(gpu_ids[0]) if use_gpu else "cpu")
    return {'device': device, 'gpu_ids': gpu_ids, 'use_gpu': use_gpu}

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def get_dataloader(dataset_args, dataloader_args):
    dataset = CityscapesDataset(**dataset_args)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_args)
    return dataloader

#### Helper functions from https://github.com/iArunava/ENet-Real-Time-Semantic-Segmentation ####

def get_class_weights(loader, num_classes, c=1.02):
    '''
    This class return the class weights for each class

    Arguments:
    - loader : The generator object which return all the labels at one iteration
               Do Note: That this class expects all the labels to be returned in
               one iteration
    - num_classes : The number of classes
    Return:
    - class_weights : An array equal in length to the number of classes
                      containing the class weights for each class
    '''
    
    labels = next(loader)
    all_labels = labels.flatten()
    each_class = np.bincount(all_labels, minlength=num_classes)
    prospensity_score = each_class / len(all_labels)
    class_weights = 1 / (np.log(c + prospensity_score))
    return class_weights


def show_images(images, in_row=True):
    '''
    Helper function to show 3 images
    '''
    total_images = len(images)
    
    rc_tuple = (1, total_images)
    if not in_row:
        rc_tuple = (total_images, 1)

    # figure = plt.figure(figsize=(20, 10))
    for ii in range(len(images)):
        plt.subplot(*rc_tuple, ii + 1)
        plt.title(images[ii][0])
        plt.axis('off')
        plt.imshow(images[ii][1])
    plt.show()
    
def decode_segmap(image):
    Sky = [128, 128, 128]
    Building = [128, 0, 0]
    Pole = [192, 192, 128]
    Road_marking = [255, 69, 0]
    Road = [128, 64, 128]
    Pavement = [60, 40, 222]
    Tree = [128, 128, 0]
    SignSymbol = [192, 128, 128]
    Fence = [64, 64, 128]
    Car = [64, 0, 128]
    Pedestrian = [64, 64, 0]
    Bicyclist = [0, 128, 192]

    label_colors = np.array([Sky, Building, Pole, Road_marking, Road,
                              Pavement, Tree, SignSymbol, Fence, Car,
                              Pedestrian, Bicyclist]).astype(np.uint8)

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for label in range(len(label_colors)):
            r[image == label] = label_colors[label, 0]
            g[image == label] = label_colors[label, 1]
            b[image == label] = label_colors[label, 2]

    rgb = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb


def loader(segmented_path, batch_size, h=512, w=512):
    """
    The Loader to generate inputs and labels from the Image and Segmented Directory
    Arguments:
    training_path - str - Path to the directory that contains the training images
    segmented_path - str - Path to the directory that contains the segmented images
    batch_size - int - the batch size
    yields inputs and labels of the batch size
    """
    
    filenames_s = os.listdir(segmented_path)
    total_files_s = len(filenames_s)
    
    if str(batch_size).lower() == 'all':
        batch_size = total_files_s
    
    idx = 0
    while (1):
        batch_idxs = np.random.randint(0, total_files_s, batch_size)
        
        labels = []
        
        for jj in batch_idxs:
            img = Image.open(segmented_path + filenames_s[jj])
            img = np.array(img)
            img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)
            labels.append(img)
        
        labels = torch.tensor(labels)
        
        yield labels
    