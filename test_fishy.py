import bdlb
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

fs = bdlb.load(benchmark="fishyscapes")
# automatically downloads the dataset
data = fs.get_dataset('Static')


# test your method with the benchmark metrics
def estimator(image):
    """Assigns a random uncertainty per pixel."""
    uncertainty = tf.random.uniform(image.shape[:-1])
    return uncertainty


def get_images(tfdataset):
    for i, blob in enumerate(tfdataset):
        image = blob['image_left'].numpy()
        mask = blob['mask'].numpy()
        Image.fromarray(image).save('/home/giancarlo/data/innosuisse/fs_static/original/image_%i.png' %i)
        cv2.imwrite('/home/giancarlo/data/innosuisse/fs_static/label/image_%i.png' %i, mask)



#metrics = fs.evaluate(estimator, data.take(2))
#print('My method achieved {:.2f}% AP'.format(100 * metrics['AP']))

get_images(data)