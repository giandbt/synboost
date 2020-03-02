import numpy as np

class DenormalizeImage(object):
    """Denormalize image based on imagenet values

    Returns:
        img: Denormalized image
    """
    def __init__(self):
        """
        """
        self.mean = np.array([0.5, 0.5, 0.5]).reshape(-1,1,1)
        self.std = np.array([0.5, 0.5, 0.5]).reshape(-1,1,1)

    def __call__(self, img):
        """
        """
        img = self.std * img + self.mean
        img = np.clip(img, 0, 1)
        return img