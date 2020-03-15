import os
import numpy as np
from PIL import Image
import cv2
import shutil

from natsort import natsorted

import sys
sys.path.append("..")
import data.cityscapes_labels as cityscapes_labels

trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid

def include_void_to_labels(label_path, semantic_path, save_dir, include_ego_vehicle = True):

    if include_ego_vehicle:
        void_labels = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30]
        if not os.path.isdir(os.path.join(save_dir, 'labels_with_void')):
            os.mkdir(os.path.join(save_dir, 'labels_with_void'))
    else:
        void_labels = [0, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30]  # without ego vehicle
        if not os.path.isdir(os.path.join(save_dir, 'labels_with_void_no_ego')):
            os.mkdir(os.path.join(save_dir, 'labels_with_void_no_ego'))

    label_paths = [os.path.join(label_path, image)
                   for image in os.listdir(label_path)]

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
        if include_ego_vehicle:
            mask_img.save(os.path.join(save_dir, 'labels_with_void', label_name))
        else:
            mask_img.save(os.path.join(save_dir, 'labels_with_void_no_ego', label_name))

def create_void_semantic(semantic_path, labels_path, semantic_path_pred, save_dir_semantic, save_dir_semantic_train, save_dir_inst):
    if not os.path.isdir(save_dir_semantic):
        os.mkdir(save_dir_semantic)
    if not os.path.isdir(save_dir_semantic_train):
        os.mkdir(save_dir_semantic_train)
    if not os.path.isdir(save_dir_inst):
        os.mkdir(save_dir_inst)

    semantic_paths = [os.path.join(semantic_path, image)
                      for image in os.listdir(semantic_path)]
    label_paths = [os.path.join(labels_path, image)
                      for image in os.listdir(labels_path)]
    pred_paths = [os.path.join(semantic_path_pred, image)
                      for image in os.listdir(semantic_path_pred)]

    semantic_paths = natsorted(semantic_paths)
    pred_paths = natsorted(pred_paths)
    label_paths = natsorted(label_paths)

    for idx, (label, semantic, pred) in enumerate(zip(label_paths, semantic_paths, pred_paths)):
        print('Generating image %i our of %i' % (idx + 1, len(label_paths)))
        new_semantic_name = os.path.basename(semantic).replace('fakeTrainIds', 'labelIds')
        new_instance_name = os.path.basename(semantic).replace('fakeTrainIds', 'instanceIds')

        semantic_train = np.array(Image.open(os.path.join(semantic_path, semantic)))
        label_img = Image.open(os.path.join(labels_path, label))
        pred_img = Image.open(os.path.join(semantic_path_pred, pred))

        pred_img = np.array(pred_img.resize(label_img.size, Image.NEAREST))
        label_img = np.array(label_img)

        semantic_out = np.zeros_like(semantic_train)
        for label_id, train_id in id_to_trainid.items():
            semantic_out[np.where(semantic_train == train_id)] = label_id

        syn_semantic = np.copy(pred_img)
        mask = (label_img==1)
        syn_semantic[mask] = semantic_out[mask]

        #syn_semantic = Image.fromarray(syn_semantic).resize(pred_img_og.size)
        #syn_semantic.save(os.path.join(save_dir_semantic, new_semantic_name))
        #syn_semantic.save(os.path.join(save_dir_inst, new_instance_name))
        cv2.imwrite(os.path.join(save_dir_semantic, new_semantic_name), syn_semantic)
        cv2.imwrite(os.path.join(save_dir_inst, new_instance_name), syn_semantic)
        semantic_out = np.zeros_like(semantic_train)
        for label_id, train_id in id_to_trainid.items():
            semantic_out[np.where(syn_semantic == label_id)] = train_id
        cv2.imwrite(os.path.join(save_dir_semantic_train, new_semantic_name), semantic_out)

def change_name(semantic_path_pred, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    pred_paths = [os.path.join(semantic_path_pred, image)
                      for image in os.listdir(semantic_path_pred)]

    pred_paths = natsorted(pred_paths)

    for idx, pred in enumerate(pred_paths):
        print('Generating image %i our of %i' % (idx + 1, len(pred_paths)))
        new_semantic_name = os.path.basename(pred).replace('_leftImg8bit', '_labelIds')
        new_instance_name = os.path.basename(pred).replace('_leftImg8bit', '_instanceIds')

        original = os.path.join(semantic_path_pred, pred)
        target1 = os.path.join(save_dir, new_semantic_name)
        target2 = os.path.join(save_dir, new_instance_name)
        shutil.copyfile(original, target1)
        shutil.copyfile(original, target2)

def create_void_semantic_original(semantic_path, semantic_path_pred, save_dir_semantic, save_dir_semantic_train, save_dir_inst):
    if not os.path.isdir(save_dir_semantic):
        os.mkdir(save_dir_semantic)
    if not os.path.isdir(save_dir_semantic_train):
        os.mkdir(save_dir_semantic_train)
    if not os.path.isdir(save_dir_inst):
        os.mkdir(save_dir_inst)

    semantic_paths = [os.path.join(semantic_path, image)
                      for image in os.listdir(semantic_path)]
    pred_paths = [os.path.join(semantic_path_pred, image)
                      for image in os.listdir(semantic_path_pred)]

    semantic_paths = natsorted(semantic_paths)
    pred_paths = natsorted(pred_paths)

    for idx, (semantic, pred) in enumerate(zip(semantic_paths, pred_paths)):
        print('Generating image %i our of %i' % (idx + 1, len(semantic_paths)))
        new_semantic_name = os.path.basename(semantic).replace('fakeTrainIds', 'labelIds')
        new_instance_name = os.path.basename(semantic).replace('fakeTrainIds', 'instanceIds')

        semantic_train = Image.open(os.path.join(semantic_path, semantic))
        pred_img = Image.open(os.path.join(semantic_path_pred, pred))
        pred_img = np.array(pred_img.resize(semantic_train.size, Image.NEAREST))
        semantic_train = np.array(semantic_train)

        semantic_out = np.zeros_like(semantic_train)
        for label_id, train_id in id_to_trainid.items():
            semantic_out[np.where(semantic_train == train_id)] = label_id

        syn_semantic = np.copy(semantic_out)
        mask = (semantic_train==255)
        syn_semantic[mask] = pred_img[mask]

        cv2.imwrite(os.path.join(save_dir_semantic, new_semantic_name), syn_semantic)
        cv2.imwrite(os.path.join(save_dir_inst, new_instance_name), syn_semantic)
        semantic_out = np.zeros_like(semantic_train)
        for label_id, train_id in id_to_trainid.items():
            semantic_out[np.where(syn_semantic == label_id)] = train_id
        cv2.imwrite(os.path.join(save_dir_semantic_train, new_semantic_name), semantic_out)

if __name__ == '__main__':

    save_dir_sema = '/home/giandbt/Documents/labels_processing/semantic_label_ids'
    save_dir_sema_train = '/home/giandbt/Documents/labels_processing/semantic'
    save_dir_inst = '/home/giandbt/Documents/labels_processing/instances'
    semantic_path = '/home/giandbt/Documents/labels_processing/semantic_epfl'
    semantic_path_pred = '/home/giandbt/Documents/labels_processing/semantic_predictions'
    labels_path = '/home/giandbt/Documents/labels_processing/labels'
    #create_void_semantic(semantic_path, labels_path, semantic_path_pred, save_dir_sema, save_dir_sema_train, save_dir_inst)

    save_dir = '/home/giandbt/Documents/labels_processing/'
    semantic_path = '/home/giandbt/Documents/labels_processing/semantic_org'
    #include_void_to_labels(labels_path, semantic_path, save_dir, include_ego_vehicle=False)

    save_dir = '/home/giandbt/Documents/labels_processing/gtFine/val'
    semantic_path_pred = '/home/giandbt/Documents/labels_processing/semantic_predictions'
    #change_name(semantic_path_pred, save_dir)

    semantic_path = '/home/giandbt/Documents/labels_processing/semantic_epfl'
    semantic_path_pred = '/home/giandbt/Documents/labels_processing/semantic_predictions'
    create_void_semantic_original(semantic_path, semantic_path_pred, save_dir_sema, save_dir_sema_train,
                                  save_dir_inst)