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


if __name__ == '__main__':
    data_path = '/media/giancarlo/Samsung_T5/master_thesis/data/lost_and_found/post-process/L&F_TrainID_labels'
    #save_dir = '/media/giancarlo/Samsung_T5/master_thesis/data/lost_and_found/post-process'
    #convert_gtCoarse_to_labels(data_path, save_dir)

    #semantic_path = '/media/giancarlo/Samsung_T5/master_thesis/data/lost_and_found/post-process/semantic_labelids'
    #convert_semantic_to_trainids(semantic_path, save_dir)