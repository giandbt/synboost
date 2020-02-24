import os
import numpy as np
from PIL import Image

from natsort import natsorted

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
        ignore_mask = np.where(semantic_img == 255, 1, 0)

        mask_img = Image.fromarray((mask).astype(np.uint8))
        ignore_mask_img = Image.fromarray((ignore_mask).astype(np.uint8))
        semantic_name = os.path.basename(semantic)
        mask_img.save(os.path.join(save_dir, 'labels', semantic_name))
        ignore_mask_img.save(os.path.join(save_dir, 'ignore', semantic_name))

if __name__ == '__main__':
    data_path = '/media/giancarlo/Samsung_T5/master_thesis/data/lost_and_found/post-process/L&F_TrainID_labels'
    save_dir = '/media/giancarlo/Samsung_T5/master_thesis/data/lost_and_found/post-process'
    convert_gtCoarse_to_labels(data_path, save_dir)