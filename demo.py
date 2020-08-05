import argparse
import os
from PIL import Image
from estimator import AnomalyDetector
import numpy as np

# function for segmentations
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    """
    Colorize a segmentation mask.
    """
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

parser = argparse.ArgumentParser()
parser.add_argument('--demo_folder', type=str, default='./sample_images', help='Path to folder with images to be run.')
parser.add_argument('--save_folder', type=str, default='./results', help='Folder to where to save the results')
opts = parser.parse_args()

demo_folder = opts.demo_folder
save_folder = opts.save_folder

images = [os.path.join(demo_folder, image) for image in os.listdir(demo_folder)]
detector = AnomalyDetector(True)

# Save folders
semantic_path = os.path.join(save_folder, 'semantic')
anomaly_path = os.path.join(save_folder, 'anomaly')
synthesis_path = os.path.join(save_folder, 'synthesis')
entropy_path = os.path.join(save_folder, 'entropy')
distance_path = os.path.join(save_folder, 'distance')
perceptual_diff_path = os.path.join(save_folder, 'perceptual_diff')

os.makedirs(semantic_path, exist_ok=True)
os.makedirs(anomaly_path, exist_ok=True)
os.makedirs(synthesis_path, exist_ok=True)
os.makedirs(entropy_path, exist_ok=True)
os.makedirs(distance_path, exist_ok=True)
os.makedirs(perceptual_diff_path, exist_ok=True)

for idx, image in enumerate(images):
    basename = os.path.basename(image).replace('.jpg', '.png')
    print('Evaluating image %i out of %i'%(idx+1, len(images)))
    image = Image.open(image)
    results = detector.estimator_image(image)

    anomaly_map = results['anomaly_map'].convert('RGB')
    anomaly_map.save(os.path.join(anomaly_path,basename))

    semantic_map = colorize_mask(np.array(results['segmentation']))
    semantic_map.save(os.path.join(semantic_path,basename))

    synthesis = results['synthesis']
    synthesis.save(os.path.join(synthesis_path,basename))

    softmax_entropy = results['softmax_entropy'].convert('RGB')
    softmax_entropy.save(os.path.join(entropy_path,basename))

    softmax_distance = results['softmax_distance'].convert('RGB')
    softmax_distance.save(os.path.join(distance_path,basename))

    perceptual_diff = results['perceptual_diff'].convert('RGB')
    perceptual_diff.save(os.path.join(perceptual_diff_path,basename))

