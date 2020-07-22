import os
from PIL import Image
from natsort import natsorted
import numpy as np
import random

import sys
sys.path.append("..")
from util import visualization
import data.cityscapes_labels as cityscapes_labels

trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid
#objects_to_change = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33] # equal_prob
objects_to_change = [7, 7, 7, 8, 11, 12, 13, 20, 21, 22, 24, 24, 24, 25, 26, 26, 26, 26, 26, 26, 26, 27, 28, 32, 33] # optmized
#objects_to_change = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28, 31, 32, 33] # no person

def create_unknown_examples(instance_path, semantic_path, original_path, save_dir, visualize=False, dynamic=False):

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if not os.path.isdir(os.path.join(save_dir, 'labels')):
        os.mkdir(os.path.join(save_dir, 'labels'))

    if not os.path.isdir(os.path.join(save_dir, 'semantic_labelId')):
        os.mkdir(os.path.join(save_dir, 'semantic_labelId'))

    if not os.path.isdir(os.path.join(save_dir, 'semantic')):
        os.mkdir(os.path.join(save_dir, 'semantic'))

    if not os.path.isdir(os.path.join(save_dir, 'original')):
        os.mkdir(os.path.join(save_dir, 'original'))

    # for creating synthesis later
    if not os.path.exists(os.path.join(save_dir, 'temp')):
        os.makedirs(os.path.join(save_dir, 'temp'))
        os.makedirs(os.path.join(save_dir, 'temp', 'gtFine', 'val'))
        os.makedirs(os.path.join(save_dir, 'temp', 'leftImg8bit', 'val'))

    semantic_paths = [os.path.join(semantic_path, image)
                           for image in os.listdir(semantic_path)]
    instance_paths = [os.path.join(instance_path, image)
                           for image in os.listdir(instance_path)]
    original_paths = [os.path.join(original_path, image)
                      for image in os.listdir(original_path)]

    semantic_paths = natsorted(semantic_paths)
    instance_paths = natsorted(instance_paths)
    original_paths = natsorted(original_paths)

    for idx, (semantic, instance, original) in enumerate(zip(semantic_paths, instance_paths, original_paths)):
        print('Generating image %i our of %i' %(idx+1, len(semantic_paths)))

        semantic_img = np.array(Image.open(semantic))
        instance_img = np.array(Image.open(instance))
        original_img = Image.open(original)
        
        unique_classes = [sample for sample in np.unique(instance_img) if len(str(sample)) == 5]

        how_many = int(random.random()*len(unique_classes))

        final_mask = np.zeros(np.shape(instance_img))
        new_semantic_map = np.copy(semantic_img)

        # Make final mask by selecting each instance to replace at random
        for _ in range(how_many):
            # instance to change
            instance_idx = int(random.random()*len(unique_classes))
            instance_change = unique_classes.pop(instance_idx)

            # get mask where instance is located
            mask = np.where(instance_img==instance_change, 1, 0)

            while True:
                new_instance_idx = int(random.random()*len(objects_to_change))
                new_instance_id = objects_to_change[new_instance_idx]

                # ensure we don't replace by the same class
                if new_instance_id != int((str(instance_change)[:2])):
                    break
            np.place(new_semantic_map, mask, new_instance_id)
            final_mask += mask

        # also mark dynamic labels (Optional)
        if dynamic:
            mask = np.where(semantic_img==5, 1, 0)
            final_mask += mask

        new_semantic_name = os.path.basename(semantic).replace('labelIds', 'unknown_labelIds')
        new_semantic_train_name = os.path.basename(semantic).replace('labelIds', 'unknown_trainIds')
        new_label_name = os.path.basename(instance).replace('instanceIds', 'unknown')
        old_semantic_name = os.path.basename(semantic)
        new_original_name = os.path.basename(original).replace('leftImg8bit', 'unknown_leftImg8bit')

        mask_img = Image.fromarray((final_mask).astype(np.uint8))

        if visualize:
            if not os.path.isdir(os.path.join(save_dir, 'old_semantic')):
                os.mkdir(os.path.join(save_dir, 'old_semantic'))
                
            # Correct labels to train ID for old semantic
            semantic_copy = semantic_img.copy()
            for k, v in id_to_trainid.items():
                semantic_copy[semantic_img == k] = v
            semantic_img = semantic_copy.astype(np.uint8)

            # Correct labels to train ID for new semantic
            semantic_copy = new_semantic_map.copy()
            for k, v in id_to_trainid.items():
                semantic_copy[new_semantic_map == k] = v
            new_semantic_map =semantic_copy.astype(np.uint8)

            new_semantic_img =visualization.colorize_mask(new_semantic_map)
            old_semantic_img = visualization.colorize_mask(semantic_img)

            # save images
            mask_img.save(os.path.join(save_dir, 'labels', new_label_name))
            new_semantic_img.save(os.path.join(save_dir, 'semantic', new_semantic_name))
            original_img.save(os.path.join(save_dir, 'original', new_original_name))
            old_semantic_img.save(os.path.join(save_dir, 'old_semantic', old_semantic_name))
        else:
            new_semantic_img = Image.fromarray(new_semantic_map)

            # Correct labels to train ID for new semantic
            semantic_copy = new_semantic_map.copy()
            for k, v in id_to_trainid.items():
                semantic_copy[new_semantic_map == k] = v
            new_semantic_map =semantic_copy.astype(np.uint8)
            new_semantic_train_img = Image.fromarray(new_semantic_map)

            # save images
            mask_img.save(os.path.join(save_dir, 'labels', new_label_name))
            original_img.save(os.path.join(save_dir, 'original', new_original_name))
            new_semantic_img.save(os.path.join(save_dir, 'semantic_labelId', new_semantic_name))
            new_semantic_train_img.save(os.path.join(save_dir, 'semantic', new_semantic_train_name))

            # save images for synthesis
            original_img.save(os.path.join(save_dir, 'temp', 'leftImg8bit', 'val', new_original_name))
            new_semantic_img.save(os.path.join(save_dir, 'temp', 'gtFine', 'val', new_semantic_name))

            instance_img = Image.open(instance)
            instance_img.save(os.path.join(save_dir, 'temp', 'gtFine', 'val', os.path.basename(instance)))

def create_known_examples(instance_path, semantic_path, original_path, save_dir):

    if not os.path.isdir(os.path.join(save_dir, 'labels')):
        os.mkdir(os.path.join(save_dir, 'labels'))

    if not os.path.isdir(os.path.join(save_dir, 'semantic')):
        os.mkdir(os.path.join(save_dir, 'semantic'))
        
    if not os.path.isdir(os.path.join(save_dir, 'original')):
        os.mkdir(os.path.join(save_dir, 'original'))

    instance_paths = [os.path.join(instance_path, image)
                      for image in os.listdir(instance_path)]

    semantic_paths = [os.path.join(semantic_path, image)
                      for image in os.listdir(semantic_path)]

    original_paths = [os.path.join(original_path, image)
                      for image in os.listdir(original_path)]

    instance_paths = natsorted(instance_paths)
    semantic_paths = natsorted(semantic_paths)
    original_paths = natsorted(original_paths)

    for idx, (instance, semantic, original) in enumerate(zip(instance_paths, semantic_paths, original_paths)):
        print('Generating image %i our of %i' % (idx + 1, len(instance_paths)))

        # create a file where all the images are zero
        instance_img = np.array(Image.open(instance))
        final_mask = np.zeros(np.shape(instance_img))

        mask_img = Image.fromarray((final_mask * 255).astype(np.uint8))
        semantic_img =Image.open(semantic)
        original_img = Image.open(original)
        
        new_semantic_name = os.path.basename(semantic).replace('labelIds', 'known_labelIds')
        new_original_name = os.path.basename(original).replace('leftImg8bit', 'known_leftImg8bit')
        label_name = os.path.basename(instance).replace('instanceIds', 'known')
        
        mask_img.save(os.path.join(save_dir, 'labels', label_name))
        semantic_img.save(os.path.join(save_dir, 'semantic', new_semantic_name))
        original_img.save(os.path.join(save_dir, 'original', new_original_name))


if __name__ == '__main__':
    instance_path = '/home/giancarlo/data/innosuisse/cityscapes/train/instances'
    semantic_path = '/home/giancarlo/data/innosuisse/cityscapes/train/semantic'
    original_path = '/home/giancarlo/data/innosuisse/cityscapes/train/original'
    save_dir = '/home/giancarlo/data/innosuisse/optimized_only'
    
    create_unknown_examples(instance_path, semantic_path, original_path, save_dir, visualize=False)
    #create_known_examples(instance_path, semantic_path, original_path, save_dir)