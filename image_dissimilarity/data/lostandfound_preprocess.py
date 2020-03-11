import os
import numpy as np
from PIL import Image

from natsort import natsorted

import sys
sys.path.append("..")
import data.cityscapes_labels as cityscapes_labels

trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid

def convert_gtCoarse_to_labels(data_path, save_dir):
    if not os.path.isdir(os.path.join(save_dir, 'labels')):
        os.mkdir(os.path.join(save_dir, 'labels'))

    semantic_paths = [os.path.join(data_path, image)
                      for image in os.listdir(data_path) if 'labelTrainIds' in image]

    semantic_paths = natsorted(semantic_paths)

    for idx, semantic in enumerate(semantic_paths):
        print('Generating image %i our of %i' % (idx + 1, len(semantic_paths)))

        semantic_img = np.array(Image.open(semantic))

        # get mask where instance is located
        mask = np.where(semantic_img == 2, 1, 0)

        mask_img = Image.fromarray((mask).astype(np.uint8))
        semantic_name = os.path.basename(semantic)
        mask_img.save(os.path.join(save_dir, 'labels', semantic_name))
        
def convert_semantic_to_trainids(semantic_path, save_dir):
    if not os.path.isdir(os.path.join(save_dir, 'semantic')):
        os.mkdir(os.path.join(save_dir, 'semantic'))

    semantic_paths = [os.path.join(semantic_path, image)
                      for image in os.listdir(semantic_path)]

    semantic_paths = natsorted(semantic_paths)

    for idx, semantic in enumerate(semantic_paths):
        print('Generating image %i our of %i' % (idx + 1, len(semantic_paths)))

        semantic_img = np.array(Image.open(semantic))

        # Correct labels to train ID
        semantic_copy = semantic_img.copy()
        for k, v in id_to_trainid.items():
            semantic_copy[semantic_img == k] = v

        semantic_img = Image.fromarray(semantic_copy.astype(np.uint8))

        semantic_name = os.path.basename(semantic)
        semantic_img.save(os.path.join(save_dir, 'semantic', semantic_name))
        
def convert_gtCoarse_to_labels_ROI(data_path, save_dir):
    if not os.path.isdir(os.path.join(save_dir, 'labels_with_ROI')):
        os.mkdir(os.path.join(save_dir, 'labels_with_ROI'))

    semantic_paths = [os.path.join(data_path, image)
                      for image in os.listdir(data_path) if 'labelTrainIds' in image]

    semantic_paths = natsorted(semantic_paths)

    for idx, semantic in enumerate(semantic_paths):
        print('Generating image %i our of %i' % (idx + 1, len(semantic_paths)))

        semantic_img = np.array(Image.open(semantic))

        # get mask where instance is located
        mask_unknown = np.where(semantic_img == 2, 1, 0)
        mask_roi = np.where(semantic_img == 255, 255, 0)

        final_mask = mask_unknown + mask_roi

        mask_img = Image.fromarray((final_mask).astype(np.uint8))
        semantic_name = os.path.basename(semantic)
        mask_img.save(os.path.join(save_dir, 'labels_with_ROI', semantic_name))
        
def include_void_to_labels(data_path, semantic_path, save_dir):
    # does not include ego car
    void_labels = [0,2,3,4,5,6,9,10,14,15,16,18,29,30]
    if not os.path.isdir(os.path.join(save_dir, 'labels_with_void')):
        os.mkdir(os.path.join(save_dir, 'labels_with_void'))

    label_paths = [os.path.join(data_path, image)
                      for image in os.listdir(data_path)]

    semantic_paths = [os.path.join(semantic_path, image)
                   for image in os.listdir(semantic_path)]

    label_paths = natsorted(label_paths)
    semantic_paths = natsorted(semantic_paths)

    for idx, (label, semantic) in enumerate(zip(label_paths, semantic_paths)):
        print('Generating image %i our of %i' % (idx + 1, len(label_paths)))

        label_img = Image.open(label)
        semantic_img = np.array(Image.open(semantic).resize(label_img.size))
        label_img = np.array(label_img).astype(np.uint8)
        # get mask where instance is located
        for void_label in void_labels:
            mask_unknown = np.where(semantic_img == void_label, 1, 0).astype(np.uint8)
            label_img += mask_unknown
            
        final_mask = np.where(label_img != 0, 1, 0).astype(np.uint8)
        mask_img = Image.fromarray((final_mask).astype(np.uint8))
        label_name = os.path.basename(label)
        mask_img.save(os.path.join(save_dir, 'labels_with_void', label_name))

if __name__ == '__main__':
    data_path = '/media/giancarlo/Samsung_T5/master_thesis/data/lost_and_found/post-process/L&F_TrainID_labels'
    #save_dir = '/media/giancarlo/Samsung_T5/master_thesis/data/lost_and_found/post-process'
    #convert_gtCoarse_to_labels(data_path, save_dir)

    #semantic_path = '/media/giancarlo/Samsung_T5/master_thesis/data/lost_and_found/post-process/semantic_labelids'
    #convert_semantic_to_trainids(semantic_path, save_dir)

    save_dir = '/media/giancarlo/Samsung_T5/master_thesis/data/dissimilarity_model/epfl_our/val/'
    label_path = '/media/giancarlo/Samsung_T5/master_thesis/data/dissimilarity_model/epfl_our/val/labels'
    semantic_path = '/media/giancarlo/Samsung_T5/data/cityscapes/datasets/cityscapes/test_label_org'
    include_void_to_labels(label_path, semantic_path, save_dir)