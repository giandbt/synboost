import bdlb
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

fs = bdlb.load(benchmark="fishyscapes")
# automatically downloads the dataset
data = fs.get_dataset('LostAndFound')


# test your method with the benchmark metrics
def estimator(image):
    """Assigns a random uncertainty per pixel."""
    uncertainty = tf.random.uniform(image.shape[:-1])
    return uncertainty


def visualize_tfdataset(tfdataset, num_samples):
    """Visualizes `num_samples` from the `tfdataset`."""
    
    fig, axs = plt.subplots(num_samples, 2, figsize=(15, 4 * num_samples))
    for i, blob in enumerate(tfdataset.take(num_samples)):
        image = blob['image_left'].numpy()
        mask = blob['mask'].numpy()
        axs[i][0].imshow(image.astype('int'))
        axs[i][0].axis("off")
        axs[i][0].set(title="Image")
        # map 255 to 2 such that difference between labels is better visible
        mask[mask == 255] = 2
        axs[i][1].imshow(mask[..., 0])
        axs[i][1].axis("off")
        axs[i][1].set(title="Mask")
    fig.show()

metrics = fs.evaluate(estimator, data.take(2))
print('My method achieved {:.2f}% AP'.format(100 * metrics['AP']))

visualize_tfdataset(data, 5)