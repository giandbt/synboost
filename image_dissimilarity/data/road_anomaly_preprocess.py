import os
import numpy as np
from PIL import Image

from natsort import natsorted

import sys
sys.path.append("..")
import data.cityscapes_labels as cityscapes_labels

trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid

def move_labels(label_path, save_dir):
    if not os.path.isdir(os.path.join(save_dir, 'labels')):
        os.mkdir(os.path.join(save_dir, 'labels'))

    image_paths = [image[:-4] for image in os.listdir(label_path) if '.jpg' in image]
    
    for image in image_paths:
        folder_name = image + '.labels'
        label_name = 'labels_semantic.png'
        label = os.path.join(label_path, folder_name, label_name)
        
        label_img = Image.open(label)
        label_img.save(os.path.join(save_dir, 'labels', image + '.png'))

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
        
def convert_labels(data_path, save_dir):
    if not os.path.isdir(os.path.join(save_dir, 'labels')):
        os.mkdir(os.path.join(save_dir, 'labels'))

    label_paths = [os.path.join(data_path, image)
                      for image in os.listdir(data_path)]

    label_paths = natsorted(label_paths)

    for idx, label in enumerate(label_paths):
        print('Generating image %i our of %i' % (idx + 1, len(label_paths)))

        label_img = np.array(Image.open(label))

        # get mask where instance is located
        mask = np.where(label_img == 2, 1, 0)

        mask_img = Image.fromarray((mask).astype(np.uint8))
        label_name = os.path.basename(label)
        mask_img.save(os.path.join(save_dir, 'labels', label_name))

if __name__ == '__main__':
    label_path = '/media/giancarlo/Samsung_T5/master_thesis/detecting-the-unexpected/data/RoadAnomaly/RoadAnomaly_jpg/frames'
    save_dir = '/media/giancarlo/Samsung_T5/master_thesis/data/road_anomaly/'
    #move_labels(label_path, save_dir)

    semantic_path = '/media/giancarlo/Samsung_T5/master_thesis/data/road_anomaly/semantic_labelids'
    #convert_semantic_to_trainids(semantic_path, save_dir)

    data_path = '/media/giancarlo/Samsung_T5/master_thesis/data/road_anomaly/labels_original'
    convert_labels(data_path, save_dir)