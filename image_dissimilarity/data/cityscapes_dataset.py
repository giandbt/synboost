import torch.utils.data as data
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
from natsort import natsorted

import sys
sys.path.append("..")
import data.cityscapes_labels as cityscapes_labels

trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid

class CityscapesDataset(Dataset):
    
    def __init__(self, dataroot, preprocess_mode, image_set, load_size=1024, crop_size=512, no_flip=False):
        self.original_paths = [os.path.join(dataroot, 'original', image)
                               for image in os.listdir(os.path.join(dataroot, 'original'))]
        self.semantic_paths = [os.path.join(dataroot, 'semantic', image)
                               for image in os.listdir(os.path.join(dataroot, 'semantic'))]
        self.synthesis_paths = [os.path.join(dataroot, 'synthesis', image)
                                for image in os.listdir(os.path.join(dataroot, 'synthesis'))]
        self.label_paths = [os.path.join(dataroot, 'semantic', image)
                            for image in os.listdir(os.path.join(dataroot, 'semantic'))] # TODO need to replace semantic for labels
        
        # We need to sort the images to ensure all the pairs match with each other
        self.original_paths = natsorted(self.original_paths)
        self.semantic_paths = natsorted(self.semantic_paths)
        self.synthesis_paths = natsorted(self.synthesis_paths)
        self.label_paths = natsorted(self.label_paths)
        
        assert len(self.original_paths) == len(self.semantic_paths) == len(self.synthesis_paths) \
               == len(self.label_paths), \
            "Number of images in the dataset does not match with each other"
        "The #images in %s and %s do not match. Is there something wrong?"
        
        self.dataset_size = len(self.original_paths)
        self.preprocess_mode = preprocess_mode
        self.load_size = load_size
        self.crop_size = crop_size
        self.no_flip = no_flip
        self.is_train = True if image_set == 'train' else False
        
    def __getitem__(self, index):
        
        # Label Image.
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        
        params = get_params(self.preprocess_mode, label.size, self.load_size, self.crop_size)
        
        transform_label = get_transform(self.preprocess_mode, params,
                                        self.load_size, self.crop_size, method=Image.NEAREST,
                                        normalize=False, no_flip=self.no_flip, is_train=self.is_train)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = 35  # 'unknown' is label_nc from cityscapes

        # input image (real images)
        image_path = self.original_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.preprocess_mode, params,
                                        self.load_size, self.crop_size, no_flip=self.no_flip, is_train=self.is_train)
        image_tensor = transform_image(image)

        # synthetic image
        syn_image_path = self.synthesis_paths[index]
        syn_image = Image.open(syn_image_path)
        syn_image = syn_image.convert('RGB')
        syn_image_tensor = transform_image(syn_image)

        # semantic image
        semantic_path = self.semantic_paths[index]
        semantic = Image.open(semantic_path)
        
        # Correct labels to train ID
        semantic = np.array(semantic)
        semantic_copy = semantic.copy()
        for k, v in id_to_trainid.items():
            semantic_copy[semantic == k] = v
        semantic = Image.fromarray(semantic_copy.astype(np.uint8))
        
        semantic_tensor = transform_label(semantic) * 255.0
        semantic_tensor[semantic_tensor == 255] = 35  # 'unknown' is label_nc from cityscapes


        input_dict = {'label': label_tensor,
                      'original': image_tensor,
                      'semantic': semantic_tensor,
                      'synthesis': syn_image_tensor,
                      'label_path': label_path,
                      'original_path': image_path,
                      'semantic_path': semantic_path,
                      'syn_image_path': syn_image_path,
                      }

        return input_dict
        
    def __len__(self):
        return self.dataset_size
    
def get_params(preprocess_mode, size, load_size = 1024, crop_size = 512):
    w, h = size
    new_h = h
    new_w = w
    if preprocess_mode == 'resize_and_crop':
        new_h = new_w = load_size
    elif preprocess_mode == 'scale_width_and_crop':
        new_w = load_size
        new_h = load_size * h // w
    elif preprocess_mode == 'scale_shortside_and_crop':
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        ls = int(load_size * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(preprocess_mode, params, load_size, crop_size,
                  aspect_ratio=1, method=Image.BICUBIC, normalize=True, toTensor=True, no_flip=False, is_train=True):
    transform_list = []
    if 'resize' in preprocess_mode:
        osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, interpolation=method))
    elif 'scale_width' in preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, load_size, method)))
    elif 'scale_shortside' in preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, load_size, method)))

    if 'crop' in preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], crop_size)))

    if preprocess_mode == 'none':
        base = 32
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if preprocess_mode == 'fixed':
        w = crop_size
        h = round(crop_size / aspect_ratio)
        transform_list.append(transforms.Lambda(lambda img: __resize(img, w, h, method)))

    if is_train and not no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

# ----------- FOR TESTING --------------

def test(dataset_args, dataloader_args, save_imgs=False, path='./visualization'):
    if save_imgs and not os.path.exists(path):
        os.makedirs(path)

    dataset = CityscapesDataset(**dataset_args)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_args)
    for counter, sample in enumerate(dataloader):
        print('Images Saved: ', sample['original'].shape[0] * counter)
        if save_imgs:
            transform = ToPILImage()
            decoder = DenormalizeImage()
            for idx, (original, label, semantic, synthesis) in \
            enumerate(zip(sample['original'], sample['label'], sample['semantic'], sample['synthesis'])):
                
                # get original image
                original = original.squeeze().cpu().numpy()
                original = torch.tensor(decoder(original), dtype=torch.float32)
                original = np.asarray(transform(original))
                original = Image.fromarray(original)
                original.save(os.path.join(path, 'Original_%i_%i' % (counter, idx) + '.png'))

                # get label image
                label = label.squeeze().cpu().numpy()
                label = np.asarray(transform(label))
                label = Image.fromarray(label).convert('RGB')
                label.save(os.path.join(path, 'Label_%i_%i' % (counter, idx) + '.png'))

                # get semantic image
                semantic = semantic.squeeze().cpu().numpy()
                semantic = np.asarray(transform(semantic))
                semantic = visualization.colorize_mask(semantic)
                semantic = semantic.convert('RGB')
                semantic.save(os.path.join(path, 'Semantic_%i_%i' % (counter, idx) + '.png'))

                # get original image
                synthesis = synthesis.squeeze().cpu().numpy()
                synthesis = torch.tensor(decoder(synthesis), dtype=torch.float32)
                synthesis = np.asarray(transform(synthesis))
                synthesis = Image.fromarray(synthesis)
                synthesis.save(os.path.join(path, 'Synthesis_%i_%i' % (counter, idx) + '.png'))
        

if __name__ == '__main__':
    from torchvision.transforms import ToPILImage
    import torch
    
    import sys
    sys.path.append("..")
    from util.image_decoders import DenormalizeImage
    from util import visualization
    
    dataset_args = {
        'dataroot': '/media/giancarlo/Samsung_T5/master_thesis/results/Pipeline',
        'preprocess_mode': 'scale_width_and_crop',
        'load_size': 1024,
        'crop_size': 512,
        'image_set': 'train',
        'no_flip': False
    }
    
    dataloader_args = {
        'batch_size': 8,
        'num_workers': 1,
        'shuffle': False
    }
    
    test(dataset_args, dataloader_args, save_imgs=True)

