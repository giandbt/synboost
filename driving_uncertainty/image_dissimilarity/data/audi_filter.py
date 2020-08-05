import os
import cv2
import numpy as np
from shutil import copyfile

hextorgb = {'ccff99': [204, 255, 153], 'ff0080': [255, 0, 128], 'b97a57': [185, 122, 87]}
labeltohex = {'animals': 'ccff99', 'obstacles': 'ff0080', 'road_blocks': 'b97a57'}


def extract_image_file_name_from_semantic_file_name(file_name_semantic):
    file_name_image = file_name_semantic.split('/')
    file_name_image = file_name_image[-1].split('.')[0]
    file_name_image = file_name_image.split('_')
    file_name_image = file_name_image[0] + '_' + \
                      'camera_' + \
                      file_name_image[2] + '_' + \
                      file_name_image[3] + '.png'
    
    return file_name_image


def filter_images(root_dir, labels_to_filter, thrd=0.01):
    root_dirs = [os.path.join(root_dir, dir) for dir in os.listdir(root_dir)]
    for jdx, dir in enumerate(root_dirs):
        if not os.path.isdir(dir):
            continue
        
        original_save_dir = os.path.join(root_dir, 'filter', 'original')
        label_save_dir = os.path.join(root_dir, 'filter', 'labels')
        
        semantic_dir = os.path.join(dir, 'label', 'cam_front_center')
        image_dir = os.path.join(dir, 'camera', 'cam_front_center')
        
        if not os.path.exists(original_save_dir):
            os.makedirs(original_save_dir)
        
        if not os.path.exists(label_save_dir):
            os.makedirs(label_save_dir)
        
        semntic_paths = [os.path.join(semantic_dir, semantic) for semantic in os.listdir(semantic_dir)]
        
        for idx, semantic_path in enumerate(semntic_paths):
            print('Processing image %i out of %i for folder %i out %i' % (
                idx + 1, len(semntic_paths), jdx + 1, len(root_dirs)))
            semantic_img = cv2.imread(semantic_path)
            semantic_img = cv2.cvtColor(semantic_img, cv2.COLOR_BGR2RGB)
            
            final_mask = np.zeros(semantic_img.shape[:2])
            for unknown_object in labels_to_filter:
                obs_color = np.asarray(hextorgb[labeltohex[unknown_object]])
                mask = cv2.inRange(semantic_img, obs_color, obs_color)
                final_mask += mask
            
            mask_labels, mask_counts = np.unique(final_mask, return_counts=True)
            
            if len(mask_labels) > 1 and mask_counts[1] / mask_counts[
                0] > thrd:  # ensures the obstacle is at least X percent of the image
                img_path = os.path.join(image_dir, extract_image_file_name_from_semantic_file_name(semantic_path))
                save_img_path = os.path.join(original_save_dir, os.path.basename(img_path))
                save_semantic_path = os.path.join(label_save_dir, os.path.basename(img_path))
                copyfile(img_path, save_img_path)
                cv2.imwrite(save_semantic_path, final_mask)


def clean_diss_labels(root_dir):
    labels_dir = os.path.join(root_dir, 'labels')
    save_dir = os.path.join(root_dir, 'labels_clean')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    label_paths = [os.path.join(labels_dir, label) for label in os.listdir(labels_dir)]
    
    for label_path in label_paths:
        label_img = cv2.imread(label_path)
        label_img = np.where(label_img == 255, 1, 0)
        cv2.imwrite(os.path.join(save_dir, os.path.basename(label_path)), label_img)


if __name__ == '__main__':
    #root_dir = '/home/giancarlo/data/innosuisse/camera_lidar_semantic/'
    root_dir = '/media/giancarlo/Samsung_T5/master_thesis/data/audi'
    #labels_to_filter = ['animals', 'obstacles']
    #filter_images(root_dir, labels_to_filter)
    clean_diss_labels(root_dir)
