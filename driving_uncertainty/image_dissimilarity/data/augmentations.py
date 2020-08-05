import numpy as np
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from imgaug import augmenters as iaa
from imgaug import parameters as iap

ImageFile.LOAD_TRUNCATED_IMAGES = True

# defines all the different types of transformations
class OnlyApplyBlurs:
    def __init__(self):
        self.aug = iaa.Sequential([iaa.Sometimes(0.25, iaa.OneOf([iaa.GaussianBlur(sigma=iap.Uniform(0, 3.0)),
                                                                  iaa.MotionBlur(
                                                                      k=iap.Choice([3, 7, 11, 15]), angle=0,
                                                                      direction=1)]))
                                   ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class OnlyApplyBlursMedium:
    def __init__(self):
        self.aug = iaa.Sequential([iaa.Sometimes(0.25, iaa.OneOf([iaa.GaussianBlur(sigma=iap.Uniform(0, 4.5)),
                                                                  iaa.MotionBlur(
                                                                      k=iap.Choice([11, 15, 21]), angle=0,
                                                                      direction=1)]))
                                   ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class OnlyApplyBlursStrong:
    def __init__(self):
        self.aug = iaa.Sequential([iaa.Sometimes(0.25, iaa.OneOf([iaa.GaussianBlur(sigma=iap.Uniform(0, 6.0)),
                                                                  iaa.MotionBlur(
                                                                      k=iap.Choice([15, 21, 27, 33]),
                                                                      angle=0,
                                                                      direction=1)]))
                                   ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class OnlyApplyBlursAggressive:
    def __init__(self):
        self.aug = iaa.Sequential([iaa.Sometimes(0.50, iaa.OneOf([iaa.GaussianBlur(sigma=iap.Uniform(0, 8.0)),
                                                                  iaa.MotionBlur(
                                                                      k=iap.Normal(15, 50),
                                                                      angle=iap.Normal(0, 360),
                                                                      direction=iap.Normal(-1, 1))]))
                                   ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class OnlyChangeContrast:
    def __init__(self):
        self.aug = iaa.Sequential(
            [iaa.Sometimes(0.25, iaa.OneOf([iaa.contrast.LinearContrast(alpha=iap.Choice(np.arange(0, 3, 0.5).tolist())),
                                            iaa.SigmoidContrast(gain=iap.Choice(np.arange(0, 3, 1).tolist()),
                                                                cutoff=iap.Choice(np.arange(0, 0.6, 0.10).tolist()))])),
             ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class OnlyApplyDropout:
    def __init__(self):
        self.aug = iaa.Sequential([iaa.Sometimes(0.25, iaa.OneOf([iaa.Dropout(p=(0, 0.2)),
                                                                  iaa.CoarseDropout(0.1, size_percent=0.25)]))
                                   ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class OnlyApplyDropoutMedium:
    def __init__(self):
        self.aug = iaa.Sequential([iaa.Sometimes(0.25, iaa.OneOf([iaa.Dropout(p=(0, 0.35)),
                                                                  iaa.CoarseDropout(0.15, size_percent=0.25)]))
                                   ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class OnlyApplyDropoutStrong:
    def __init__(self):
        self.aug = iaa.Sequential([iaa.Sometimes(0.40, iaa.OneOf([iaa.Dropout(p=(0, 0.5)),
                                                                  iaa.CoarseDropout(0.25, size_percent=0.25)]))
                                   ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class OnlyApplyDropoutAggressive:
    def __init__(self):
        self.aug = iaa.Sequential([iaa.Sometimes(0.50, iaa.OneOf([iaa.Dropout(p=(0, 0.75)),
                                                                  iaa.CoarseDropout(0.5, size_percent=0.5)]))
                                   ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class OnlyApplyNoiseLight:
    def __init__(self):
        self.aug = iaa.Sequential(
            [iaa.Sometimes(0.25, iaa.OneOf([iaa.AdditiveGaussianNoise((0, 0.1), (0, 0.1), per_channel=True),
                                            iaa.AdditivePoissonNoise((0, 0.1), per_channel=True)]))
             ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class OnlyApplyNoiseMedium:
    def __init__(self):
        self.aug = iaa.Sequential(
            [iaa.Sometimes(0.25, iaa.OneOf([iaa.AdditiveGaussianNoise((0, 0.2), (0, 0.1), per_channel=True),
                                            iaa.AdditivePoissonNoise((0, 0.2), per_channel=True)]))
             ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class OnlyApplyNoiseStrong:
    def __init__(self):
        self.aug = iaa.Sequential(
            [iaa.Sometimes(0.25, iaa.OneOf([iaa.AdditiveGaussianNoise((0, 0.2), (0, 0.2), per_channel=True),
                                            iaa.AdditivePoissonNoise((0, 0.2), per_channel=True)]))
             ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class OnlyApplyNoiseAggressive:
    def __init__(self):
        self.aug = iaa.Sequential(
            [iaa.Sometimes(0.25, iaa.OneOf([iaa.AdditiveGaussianNoise((0, 100),
                                                                      (0, 100),
                                                                      per_channel=True),
                                            iaa.AdditivePoissonNoise((0, 100),
                                                                     per_channel=True)]))
             ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class OnlyApplyBrightnessAggressive:
    def __init__(self):
        self.aug = iaa.Sequential([iaa.Add(iap.Normal(-200, 200), per_channel=False)])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def get_transform(image_size, transform_name='blurs'):
    # uses ImageNet mean and standard deviation to normalize images
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    my_transforms = dict()
    #common_transforms = [transforms.Normalize(norm_mean, norm_std)]
    common_transforms = [transforms.Resize(size=image_size, interpolation=Image.NEAREST),transforms.ToTensor()]
    my_transforms['none'] = []
    my_transforms['base'] = transforms.Compose(common_transforms)
    my_transforms['normalization'] = transforms.Compose(common_transforms)
    my_transforms['blurs'] = transforms.Compose([OnlyApplyBlurs(), lambda x: Image.fromarray(x)] + common_transforms)
    my_transforms['contrast'] = transforms.Compose(
        [OnlyChangeContrast(), lambda x: Image.fromarray(x)] + common_transforms)
    my_transforms['dropout'] = transforms.Compose(
        [OnlyApplyDropout(), lambda x: Image.fromarray(x)] + common_transforms)
    my_transforms['color_jitter'] = transforms.Compose([transforms.ColorJitter(0.4, 0.4, 0.4)] + common_transforms)
    my_transforms['color_jitter_dropout'] = transforms.Compose([OnlyApplyDropout(), lambda x: Image.fromarray(x)] +
                                                               [transforms.ColorJitter(0.4, 0.4, 0.4)] +
                                                               common_transforms)
    my_transforms['geometry'] = transforms.Compose(
        [transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=(-25, +25)),
                                 transforms.RandomRotation((-25, 25))], p=0.50)] + common_transforms)

    my_transforms['all'] = transforms.Compose([OnlyApplyBlurs(),
                                               OnlyChangeContrast(),
                                               OnlyApplyDropout(), lambda x: Image.fromarray(x)] +
                                              [transforms.ColorJitter(0.4, 0.4, 0.4)] +
                                              common_transforms)

    my_transforms['light_1'] = transforms.Compose([OnlyApplyDropout(),
                                                   lambda x: Image.fromarray(x)] +
                                                  [transforms.ColorJitter(0.3, 0.3, 0.3)] +
                                                  common_transforms)

    my_transforms['light_2'] = transforms.Compose([OnlyApplyBlurs(),
                                                   lambda x: Image.fromarray(x)] +
                                                  [transforms.ColorJitter(0.3, 0.3, 0.3)] +
                                                  common_transforms)

    my_transforms['light_3'] = transforms.Compose([OnlyApplyBlurs(),
                                                   OnlyApplyNoiseMedium(),
                                                   lambda x: Image.fromarray(x)] +
                                                  [transforms.ColorJitter(0.3, 0.3, 0.3)] +
                                                  common_transforms)

    my_transforms['medium_1'] = transforms.Compose([OnlyApplyDropoutMedium(),
                                                    lambda x: Image.fromarray(x)] +
                                                   [transforms.ColorJitter(0.4, 0.4, 0.4)] +
                                                   common_transforms)

    my_transforms['medium_2'] = transforms.Compose([OnlyApplyBlursMedium(),
                                                    lambda x: Image.fromarray(x)] +
                                                   [transforms.ColorJitter(0.4, 0.4, 0.4)] +
                                                   common_transforms)

    my_transforms['medium_3'] = transforms.Compose([OnlyApplyBlursMedium(),
                                                    OnlyApplyNoiseMedium(),
                                                    lambda x: Image.fromarray(x)] +
                                                   [transforms.ColorJitter(0.4, 0.4, 0.4)] +
                                                   common_transforms)

    my_transforms['medium_4'] = transforms.Compose([OnlyApplyBlursMedium(),
                                                    OnlyApplyDropoutMedium(),
                                                    OnlyApplyNoiseMedium(),
                                                    lambda x: Image.fromarray(x)] +
                                                   [transforms.ColorJitter(0.4, 0.4, 0.4)] +
                                                   common_transforms)

    my_transforms['strong_1'] = transforms.Compose([OnlyApplyBlursStrong(),
                                                    OnlyApplyDropoutStrong(),
                                                    lambda x: Image.fromarray(x)] +
                                                   [transforms.ColorJitter(0.5, 0.5, 0.5)] +
                                                   common_transforms)

    my_transforms['strong_2'] = transforms.Compose([OnlyApplyBlursStrong(),
                                                    OnlyApplyDropoutStrong(),
                                                    lambda x: Image.fromarray(x)] +
                                                   [transforms.ColorJitter(0.5, 0.5, 0.5)] +
                                                   common_transforms)

    my_transforms['strong_3'] = transforms.Compose([OnlyApplyBlursStrong(),
                                                    OnlyApplyNoiseMedium(),
                                                    lambda x: Image.fromarray(x)] +
                                                   [transforms.ColorJitter(0.5, 0.5, 0.5)] +
                                                   common_transforms)

    my_transforms['strong_4'] = transforms.Compose([OnlyApplyBlursStrong(),
                                                    OnlyApplyDropoutStrong(),
                                                    OnlyApplyNoiseMedium(),
                                                    lambda x: Image.fromarray(x)] +
                                                   [transforms.ColorJitter(0.5, 0.5, 0.5)] +
                                                   common_transforms)

    return my_transforms['base'], my_transforms[transform_name]
