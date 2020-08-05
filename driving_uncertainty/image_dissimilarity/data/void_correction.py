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
    print(include_ego_vehicle)
    # we only include void labels that are borders, static or big objects
    if include_ego_vehicle:
        #void_labels = [0, 1, 2, 3, 4,  29, 30]
        void_labels = [4]
        if not os.path.isdir(os.path.join(save_dir, 'labels_with_void')):
            os.mkdir(os.path.join(save_dir, 'labels_with_void'))
    else:
        #void_labels = [0, 2, 3, 4,  29, 30]  # without ego vehicle
        void_labels = [5]  # without ego vehicle
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
            #mask_unknown = np.where(semantic_img == void_label, 1, 0).astype(np.uint8)
            mask_unknown = np.where(semantic_img == void_label, 255, 0).astype(np.uint8)
            label_img += mask_unknown

        #final_mask = np.where(label_img != 1, 0, 1).astype(np.uint8)
        final_mask = np.where(label_img != 255, 0, 255).astype(np.uint8)
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

def change_labelIds_to_trainIds(semantic_folder,save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    semantic_paths = [os.path.join(semantic_folder, image)
                      for image in os.listdir(semantic_folder)]

    semantic_paths = natsorted(semantic_paths)

    for idx, semantic in enumerate(semantic_paths):
        print('Generating image %i our of %i' % (idx + 1, len(semantic_paths)))
        new_semantic_name = os.path.basename(semantic).replace('labelIds', 'TrainIds_known')

        semantic = np.array(Image.open(os.path.join(semantic_path, semantic)))
        semantic_out = np.zeros_like(semantic)
        for label_id, train_id in id_to_trainid.items():
            semantic_out[np.where(semantic == label_id)] = train_id

        cv2.imwrite(os.path.join(save_dir, new_semantic_name), semantic_out)
        
def change_trainIds_to_labelIds(semantic_folder,save_dir, semantic_path_pred):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    semantic_path_pred = [os.path.join(semantic_path_pred, image)
                      for image in os.listdir(semantic_path_pred)]
    semantic_paths = [os.path.join(semantic_folder, image)
                      for image in os.listdir(semantic_folder)]

    semantic_paths_pred = natsorted(semantic_path_pred)
    semantic_paths = natsorted(semantic_paths)
    for idx, (semantic_path, semantic_path_pred) in enumerate(zip(semantic_paths, semantic_paths_pred)):
        print('Generating image %i our of %i' % (idx + 1, len(semantic_paths)))
        new_semantic_name = os.path.basename(semantic_path)
        semantic = Image.open(semantic_path)
        semantic_pred = np.array(Image.open(semantic_path_pred).resize(semantic.size, Image.NEAREST))
        semantic = np.array(semantic)
        
        semantic_out = np.zeros_like(semantic)
        for label_id, train_id in id_to_trainid.items():
            semantic_out[np.where(semantic == train_id)] = label_id
            
        syn_semantic = np.copy(semantic_out)
        mask = (semantic == 255)
        syn_semantic[mask] = semantic_pred[mask]
        
        cv2.imwrite(os.path.join(save_dir, new_semantic_name), syn_semantic)

def create_labels(semantic_folder,save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    semantic_paths = [os.path.join(semantic_folder, image)
                      for image in os.listdir(semantic_folder)]

    semantic_paths = natsorted(semantic_paths)

    for idx, semantic in enumerate(semantic_paths):
        print('Generating image %i our of %i' % (idx + 1, len(semantic_paths)))
        new_semantic_name = os.path.basename(semantic).replace('trainlIds', 'label')

        semantic = np.array(Image.open(os.path.join(semantic_path, semantic)))
        semantic_out = np.zeros_like(semantic)
        cv2.imwrite(os.path.join(save_dir, new_semantic_name), semantic_out)

def create_labels_fake(semantic_folder,save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    semantic_paths = [os.path.join(semantic_folder, image)
                      for image in os.listdir(semantic_folder)]

    semantic_paths = natsorted(semantic_paths)

    for idx, semantic in enumerate(semantic_paths):
        print('Generating image %i our of %i' % (idx + 1, len(semantic_paths)))
        new_semantic_name = os.path.basename(semantic)
        semantic = np.array(Image.open(os.path.join(semantic_path, semantic)))
        semantic_out = np.ones_like(semantic)
        cv2.imwrite(os.path.join(save_dir, new_semantic_name), semantic_out)

def update_labels_to_ignore_void(semantic_path, labels_path, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    semantic_paths = [os.path.join(semantic_path, image)
                      for image in os.listdir(semantic_path)]
    labels_paths = [os.path.join(labels_path, image)
                      for image in os.listdir(labels_path)]

    semantic_paths = natsorted(semantic_paths)
    labels_paths = natsorted(labels_paths)

    for idx, (label, semantic) in enumerate(zip(labels_paths, semantic_paths)):
        print('Generating image %i our of %i' % (idx + 1, len(semantic_paths)))
        label_name = os.path.basename(label)
        semantic = np.array(Image.open(os.path.join(semantic_path, semantic)))
        label = np.array(Image.open(os.path.join(labels_path, label)))
        label[semantic == 255] = 255
        cv2.imwrite(os.path.join(save_dir, label_name), label)

if __name__ == '__main__':

    #save_dir_sema = '/home/giandbt/Documents/labels_processing/semantic_label_ids'
    #save_dir_sema_train = '/home/giandbt/Documents/labels_processing/semantic'
    #save_dir_inst = '/home/giandbt/Documents/labels_processing/instances'
    #semantic_path = '/home/giandbt/Documents/labels_processing/semantic_epfl'
    #semantic_path_pred = '/home/giandbt/Documents/labels_processing/semantic_predictions'
    #labels_path = '/home/giandbt/Documents/labels_processing/labels'
    #create_void_semantic(semantic_path, labels_path, semantic_path_pred, save_dir_sema, save_dir_sema_train, save_dir_inst)

    #save_dir = '/home/giandbt/Documents/labels_processing/'
    #semantic_path = '/home/giandbt/Documents/labels_processing/semantic_org'
    #include_void_to_labels(labels_path, semantic_path, save_dir, include_ego_vehicle=True)

    #save_dir = '/home/giandbt/Documents/labels_processing/gtFine/val'
    #semantic_path_pred = '/home/giandbt/Documents/labels_processing/semantic_predictions'
    #change_name(semantic_path_pred, save_dir)

    #semantic_path = '/home/giandbt/Documents/labels_processing/semantic_epfl'
    #semantic_path_pred = '/home/giandbt/Documents/labels_processing/semantic_predictions'
    #create_void_semantic_original(semantic_path, semantic_path_pred, save_dir_sema, save_dir_sema_train,  save_dir_inst)

    #semantic_folder = '/home/giandbt/Documents/data/master_thesis/dissimilarity_model/epfl_combined/semantic_label'
    #save_dir = '/home/giandbt/Documents/data/master_thesis/dissimilarity_model/epfl_combined/semantc'
    #change_labelIds_to_trainIds(semantic_folder, save_dir)

    #semantic_folder = '/home/giandbt/Documents/data/master_thesis/dissimilarity_model/epfl_combined/semantc'
    #save_dir = '/home/giandbt/Documents/data/master_thesis/dissimilarity_model/epfl_combined/labels'
    #create_labels(semantic_folder, save_dir)

    #semantic_folder = '/home/giandbt/Documents/data/master_thesis/dissimilarity_model/indoorCVPR_09_post-process/semantic'
    #save_dir = '/home/giandbt/Documents/data/master_thesis/dissimilarity_model/indoorCVPR_09_post-process/labels'
    #create_labels_fake(semantic_folder, save_dir)

    #semantic_path = '/home/giandbt/Documents/data/master_thesis/dissimilarity_model/epfl_clean/train/semantic'
    #labels_path = '/home/giandbt/Documents/data/master_thesis/dissimilarity_model/epfl_clean/train/labels'
    #save_dir = '/home/giandbt/Documents/data/master_thesis/dissimilarity_model/epfl_clean/train/labels_ignore'
    #update_labels_to_ignore_void(semantic_path, labels_path, save_dir)
    
    semantic_folder = '/home/giancarlo/data/innosuisse/custom_both/semantic'
    semantic_pred_folder = '/home/giancarlo/data/innosuisse/custom_both/semantic_label_ids_icnet'
    save_dir = '/home/giancarlo/data/innosuisse/custom_both/semantic_labelsId_original'
    change_trainIds_to_labelIds(semantic_folder, save_dir, semantic_pred_folder)


