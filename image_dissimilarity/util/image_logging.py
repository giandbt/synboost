import numpy as np
import torch
from torchvision.transforms import ToPILImage

class DenormalizeImage(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)

        transform = ToPILImage()
        image = transform(tensor)
        return image

class ImgLogging(object):
    """Maintain track and merge images and their predictions for visualization

    Args:
        decoders_dic (dictionary): Dictionary for decoding all outputs for plotting
    """
    def __init__(self, preprocess_mode):
        # image net normalization values
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

        self.imgs = []

        # Decoders depending on the type of augmentations done
        if preprocess_mode == 'none':
            self.decoders = {'original': ToPILImage(), 'label': ToPILImage(),
                             'synthesis': ToPILImage(), 'semantic': ToPILImage(), 'prediction': ToPILImage()}
        else:
            self.decoders = {'original': DenormalizeImage(norm_mean, norm_std), 'label': ToPILImage(),
                             'synthesis': DenormalizeImage(norm_mean, norm_std), 'semantic': ToPILImage(),
                             'prediction': ToPILImage}

    def decode_img(self, dic):
        img_labels = []
        for ind, (key, image) in enumerate(dic.items()):
            img_labels.append(np.asarray(self.decoders[key](image).convert('RGB')))
        return img_labels

    def merge(self, img_list, row=True):
        num_imgs, height, width, channels = np.shape(img_list)
        if row:
            total_width = width * num_imgs
            total_height = height
        else:
            total_height = height * num_imgs
            total_width = width

        new_im = np.zeros((3, total_height, total_width))
        if row:
            width_start = 0
            width_end = width
            height_start = 0
            height_end = total_height
        else:
            width_start = 0
            width_end = total_width
            height_start = 0
            height_end = height

        for im in img_list:
            im = np.transpose(im, (2,0,1))
            new_im[:, height_start:height_end, width_start:width_end] = im
            if row:
                width_start += width
                width_end = width_start + width
            else:
                height_start += height
                height_end = height_start + height

        return new_im

    def log_imgs(self, dic):
        self.imgs.append(self.merge(self.decode_img(dic)))

    def write_imgs(self, writer, iteration):
        import pdb; pdb.set_trace()
        writer.add_image('image', self.imgs, iteration)
        self.imgs = []

