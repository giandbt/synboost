import bdlb
import tensorflow as tf

# Error with AssertionError: Manual directory /home/giancarlo/tensorflow_datasets/downloads/manual/cityscapes does not exist. Create it and download/extract dataset artifacts in there. Additional instructions: None

fs = bdlb.load(benchmark="fishyscapes")
# automatically downloads the dataset
data = fs.get_dataset('LostAndFound')
# test your method with the benchmark metrics
def estimator(image):
    """Assigns a random uncertainty per pixel."""
    uncertainty = tf.random.uniform(image.shape[:-1])
    return uncertainty

metrics = fs.evaluate(estimator, data.take(2))
print('My method achieved {:.2f}% AP'.format(100 * metrics['AP']))