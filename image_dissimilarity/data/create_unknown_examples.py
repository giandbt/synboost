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
objects_to_change = np.arange(24,33) # instance labels from cityscapes

def create_unknown_examples(instance_path, semantic_path, save_dir, visualize=False):

    if not os.path.isdir(os.path.join(save_dir, 'labels')):
        os.mkdir(os.path.join(save_dir, 'labels'))

    if not os.path.isdir(os.path.join(save_dir, 'semantic')):
        os.mkdir(os.path.join(save_dir, 'semantic'))

    semantic_paths = [os.path.join(semantic_path, image)
                           for image in os.listdir(semantic_path)]
    instance_paths = [os.path.join(instance_path, image)
                           for image in os.listdir(instance_path)]

    semantic_paths = natsorted(semantic_paths)
    instance_paths = natsorted(instance_paths)

    for idx, (semantic, instance) in enumerate(zip(semantic_paths, instance_paths)):
        print('Generating image %i our of %i' %(idx+1, len(semantic_paths)))

        semantic_img = np.array(Image.open(semantic))
        instance_img = np.array(Image.open(instance))
        unique_classes = [sample for sample in np.unique(instance_img) if len(str(sample)) == 5]

        how_many = int(random.random()*len(unique_classes)/2) # We only change a maximum of half the instances

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
                print('Same!')
            np.place(new_semantic_map, mask, new_instance_id)
            final_mask += mask

        new_semantic_name = os.path.basename(semantic).replace('labelIds', 'labelIds_unknown')
        new_label_name = os.path.basename(instance).replace('instanceIds', 'unknown')
        old_semantic_name = os.path.basename(semantic)

        mask_img = Image.fromarray((final_mask * 255).astype(np.uint8))

        if visualize:
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
            mask_img.save(os.path.join(save_dir, new_label_name))
            new_semantic_img.save(os.path.join(save_dir, new_semantic_name))
            old_semantic_img.save(os.path.join(save_dir, old_semantic_name))
        else:
            new_semantic_img = Image.fromarray(new_semantic_map)

            # save images
            mask_img.save(os.path.join(save_dir, 'label', new_label_name))
            new_semantic_img.save(os.path.join(save_dir, 'semantic', new_semantic_name))

def create_known_examples(instance_path, save_dir):

    if not os.path.isdir(os.path.join(save_dir, 'labels')):
        os.mkdir(os.path.join(save_dir, 'labels'))

    if not os.path.isdir(os.path.join(save_dir, 'semantic')):
        os.mkdir(os.path.join(save_dir, 'semantic'))

    instance_paths = [os.path.join(instance_path, image)
                      for image in os.listdir(instance_path)]

    instance_paths = natsorted(instance_paths)

    for idx, instance in enumerate(instance_paths):
        print('Generating image %i our of %i' % (idx + 1, len(instance_paths)))

        # create a file where all the images are zero
        instance_img = np.array(Image.open(instance))
        final_mask = np.zeros(np.shape(instance_img))

        mask_img = Image.fromarray((final_mask * 255).astype(np.uint8))
        label_name = os.path.basename(instance).replace('instanceIds', 'known')
        mask_img.save(os.path.join(save_dir, 'labels', label_name))


if __name__ == '__main__':
    instance_path = '/home/giandbt/Documents/data/instances'
    semantic_path = '/home/giandbt/Documents/data/semantic'
    save_dir = '/home/giandbt/Documents/data/unknown'
    #create_unknown_examples(instance_path, semantic_path, save_dir, visualize=True)

    create_known_examples(instance_path, save_dir)