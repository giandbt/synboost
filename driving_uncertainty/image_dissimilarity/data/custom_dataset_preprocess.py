import json
import numpy as np
import os

from natsort import natsorted
from PIL import Image


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset_rec(dir, images):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dnames, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)


def make_dataset(dir, recursive=False, read_cache=False, write_cache=False):
    images = []

    if read_cache:
        possible_filelist = os.path.join(dir, 'files.list')
        if os.path.isfile(possible_filelist):
            with open(possible_filelist, 'r') as f:
                images = f.read().splitlines()
                return images

    if recursive:
        make_dataset_rec(dir, images)
    else:
        assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

        for root, dnames, fnames in sorted(os.walk(dir)):
            fnames = sorted(fnames)
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    if write_cache:
        filelist_cache = os.path.join(dir, 'files.list')
        with open(filelist_cache, 'w') as f:
            for path in images:
                f.write("%s\n" % path)
            print('wrote filelist cache at %s' % filelist_cache)

    return images
def get_mapillary_labels(data_dir, data_type, save_dir):

    # read in config file
    config_file = os.path.join(data_dir, 'config.json')
    with open(config_file) as config_file:
        config = json.load(config_file)
    # in this example we are only interested in the labels
    labels = config['labels']
    for label_id, label in enumerate(labels):
        print("{:>30} ({:2d}): {:<40} has instances: {}".format(label["readable"], label_id, label["name"],
                                                                label["instances"]))
    # labels to show
    void_labels = [0, 1, 32, 33, 35, 42, 43, 51, 53, 63, 65]
    #void_labels = [65]

     # create save folder
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # find label images
    label_path = os.path.join(data_dir, data_type, 'labels')
    label_paths = [os.path.join(label_path, image)
                      for image in os.listdir(label_path)]
    label_paths = natsorted(label_paths)

    for idx, label in enumerate(label_paths):
        print('Generating image %i our of %i' % (idx + 1, len(label_paths)))
        # load images
        label_image = Image.open(label)
        # convert labeled data to numpy arrays for better handling
        label_array = np.array(label_image)
        label_img = np.zeros(label_array.shape)
        for void_label in void_labels:
            mask_unknown = np.where(label_array == void_label, 1, 0).astype(np.uint8)
            label_img += mask_unknown

        final_mask = np.where(label_img != 1, 0, 1).astype(np.uint8)
        mask_img = Image.fromarray((final_mask).astype(np.uint8))
        label_name = os.path.basename(label)
        mask_img.save(os.path.join(save_dir, label_name))


def get_cityscapes_labels(data_dir, data_type, save_dir):

    # labels to show
    void_labels = [5]

    # create save folder
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # find label images
    label_dir = os.path.join(data_dir, 'gtFine', data_type)
    label_paths_all = make_dataset(label_dir, recursive=True)
    label_paths = [p for p in label_paths_all if p.endswith('_labelIds.png')]
    label_paths = natsorted(label_paths)

    for idx, label in enumerate(label_paths):
        print('Generating image %i our of %i' % (idx + 1, len(label_paths)))
        # load images
        label_image = Image.open(label)
        # convert labeled data to numpy arrays for better handling
        label_array = np.array(label_image)
        label_img = np.zeros(label_array.shape)
        for void_label in void_labels:
            mask_unknown = np.where(label_array == void_label, 1, 0).astype(np.uint8)
            label_img += mask_unknown

        final_mask = np.where(label_img != 1, 0, 1).astype(np.uint8)
        if len(np.unique(final_mask)) == 2:
            mask_img = Image.fromarray((final_mask).astype(np.uint8))
            label_name = os.path.basename(label)
            mask_img.save(os.path.join(save_dir, label_name))

def get_wild_dash_labels(data_dir, save_dir):

    # labels to show
    void_labels = [5]

    # create save folder
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # find label images
    label_dir = os.path.join(data_dir)
    label_paths_all = make_dataset(label_dir, recursive=True)
    label_paths = [p for p in label_paths_all if p.endswith('_labelIds.png')]
    label_paths = natsorted(label_paths)

    for idx, label in enumerate(label_paths):
        print('Generating image %i our of %i' % (idx + 1, len(label_paths)))
        # load images
        label_image = Image.open(label)
        # convert labeled data to numpy arrays for better handling
        label_array = np.array(label_image)
        label_img = np.zeros(label_array.shape)
        for void_label in void_labels:
            mask_unknown = np.where(label_array == void_label, 1, 0).astype(np.uint8)
            label_img += mask_unknown

        final_mask = np.where(label_img != 1, 0, 255).astype(np.uint8)
        mask_img = Image.fromarray((final_mask).astype(np.uint8))
        label_name = os.path.basename(label)
        mask_img.save(os.path.join(save_dir, label_name))


def get_cityscapes_labels_dynamic(data_dir, data_type, save_dir):

    # labels to show
    void_labels = [5]

    # create save folder
    original_path = os.path.join(save_dir, 'original')
    semantic_path = os.path.join(save_dir, 'semantic')
    synthesis_path = os.path.join(save_dir, 'synthesis')
    label_save_path = os.path.join(save_dir, 'labels')

    # find label images
    label_dir = os.path.join(data_dir, 'gtFine', data_type)
    label_paths_all = make_dataset(label_dir, recursive=True)
    label_paths = [p for p in label_paths_all if p.endswith('_labelIds.png')]

    semantic_paths = [os.path.join(semantic_path, image)
                      for image in os.listdir(semantic_path)]

    synthesis_paths = [os.path.join(synthesis_path, image)
                      for image in os.listdir(synthesis_path)]

    original_paths = [os.path.join(original_path, image)
                      for image in os.listdir(original_path)]

    label_paths = natsorted(label_paths)
    original_paths = natsorted(original_paths)
    synthesis_paths = natsorted(synthesis_paths)
    semantic_paths = natsorted(semantic_paths)

    for idx, (label, original, semantic, synthesis) in enumerate(zip(label_paths, original_paths, semantic_paths, synthesis_paths)):
        print('Generating image %i our of %i' % (idx + 1, len(label_paths)))
        # load images
        label_image = Image.open(label)
        # convert labeled data to numpy arrays for better handling
        label_array = np.array(label_image)
        label_img = np.zeros(label_array.shape)
        for void_label in void_labels:
            mask_unknown = np.where(label_array == void_label, 1, 0).astype(np.uint8)
            label_img += mask_unknown

        final_mask = np.where(label_img != 1, 0, 1).astype(np.uint8)
        if len(np.unique(final_mask)) == 2:
            mask_img = Image.fromarray((final_mask).astype(np.uint8))
            label_name = os.path.basename(label)
            mask_img.save(os.path.join(label_save_path, label_name))
        else:
            os.remove(original)
            os.remove(semantic)
            os.remove(synthesis)

if __name__ == "__main__":
    #data_dir = '/media/giandbt/Samsung_T5/data/Mapillary_Vistas'
    #data_type = 'validation'
    #save_dir = '/home/giandbt/Documents/mapillary_postprocess/labels'
    #get_mapillary_labels(data_dir, data_type, save_dir)

    #data_dir = '/media/giandbt/Samsung_T5/data/cityscapes'
    #data_type = 'train'
    #save_dir = '/home/giandbt/Documents/cityscapes_postprocess/labels'
    #get_cityscapes_labels(data_dir, data_type, save_dir)

    #data_dir = '/home/giandbt/Documents/data/master_thesis/wild_dash'
    #save_dir = '/home/giandbt/Documents/wilddash_postprocess/labels'
    #get_wild_dash_labels(data_dir, save_dir)

    data_dir = '/media/giandbt/Samsung_T5/data/cityscapes'
    data_type = 'train'
    save_dir = '/home/giandbt/Desktop/custom_dynamic'
    get_cityscapes_labels_dynamic(data_dir, data_type, save_dir)