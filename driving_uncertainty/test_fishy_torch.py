import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import yaml
import random
from driving_uncertainty.options.config_class import Config

import sys
sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(__file__), 'image_segmentation'))
import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg
sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(__file__), 'image_synthesis'))
from driving_uncertainty.image_synthesis.models.pix2pix_model import Pix2PixModel
from driving_uncertainty.image_dissimilarity.models.dissimilarity_model import DissimNetPrior, DissimNet
from driving_uncertainty.image_dissimilarity.models.vgg_features import VGG19_difference
from driving_uncertainty.image_dissimilarity.data.cityscapes_dataset import one_hot_encoding


class AnomalyDetector():
    def __init__(self, ours=True, seed=0):
        
        self.set_seeds(seed)
        
        # Common options for all models
        TestOptions = Config()
        self.opt = TestOptions
        torch.cuda.empty_cache()
        self.get_segmentation()
        self.get_synthesis()
        self.get_dissimilarity(ours)
        self.get_transformations()

    # Loop around all figures
    def estimator_worker(self, image):
        image_og_h = image.shape[0]
        image_og_w = image.shape[1]
        img = Image.fromarray(np.array(image)).convert('RGB').resize((2048, 1024))
        img_tensor = self.img_transform(img)
        
        # predict segmentation
        with torch.no_grad():
            seg_outs = self.seg_net(img_tensor.unsqueeze(0).cuda())
        
        seg_softmax_out = F.softmax(seg_outs, dim=1)
        seg_final = np.argmax(seg_outs.cpu().numpy().squeeze(), axis=0)  # segmentation map
        
        # get entropy
        entropy = torch.sum(-seg_softmax_out * torch.log(seg_softmax_out), dim=1)
        entropy = (entropy - entropy.min()) / entropy.max()
        entropy *= 255  # for later use in the dissimilarity
        
        # get softmax distance
        distance, _ = torch.topk(seg_softmax_out, 2, dim=1)
        max_logit = distance[:, 0, :, :]
        max2nd_logit = distance[:, 1, :, :]
        result = max_logit - max2nd_logit
        distance = 1 - (result - result.min()) / result.max()
        distance *= 255  # for later use in the dissimilarity
        
        # get label map for synthesis model
        label_out = np.zeros_like(seg_final)
        for label_id, train_id in self.opt.dataset_cls.id_to_trainid.items():
            label_out[np.where(seg_final == train_id)] = label_id
        label_img = Image.fromarray((label_out).astype(np.uint8))
        
        # prepare for synthesis
        label_tensor = self.transform_semantic(label_img) * 255.0
        label_tensor[label_tensor == 255] = 35  # 'unknown' is opt.label_nc
        image_tensor = self.transform_image_syn(img)
        # Get instance map in right format. Since prediction doesn't have instance map, we use semantic instead
        instance_tensor = label_tensor.clone()
        
        # run synthesis
        syn_input = {'label': label_tensor.unsqueeze(0), 'instance': instance_tensor.unsqueeze(0),
                     'image': image_tensor.unsqueeze(0)}
        generated = self.syn_net(syn_input, mode='inference')
        
        image_numpy = (np.transpose(generated.squeeze().cpu().numpy(), (1, 2, 0)) + 1) / 2.0
        synthesis_final_img = Image.fromarray((image_numpy * 255).astype(np.uint8))
        
        # prepare dissimilarity
        entropy = entropy.cpu().numpy()
        distance = distance.cpu().numpy()
        entropy_img = Image.fromarray(entropy.astype(np.uint8).squeeze())
        distance = Image.fromarray(distance.astype(np.uint8).squeeze())
        semantic = Image.fromarray((seg_final).astype(np.uint8))
        
        # get initial transformation
        semantic_tensor = self.base_transforms_diss(semantic) * 255
        syn_image_tensor = self.base_transforms_diss(synthesis_final_img)
        image_tensor = self.base_transforms_diss(img)
        syn_image_tensor = self.norm_transform_diss(syn_image_tensor).unsqueeze(0).cuda()
        image_tensor = self.norm_transform_diss(image_tensor).unsqueeze(0).cuda()
        
        # get softmax difference
        perceptual_diff = self.vgg_diff(image_tensor, syn_image_tensor)
        min_v = torch.min(perceptual_diff.squeeze())
        max_v = torch.max(perceptual_diff.squeeze())
        perceptual_diff = (perceptual_diff.squeeze() - min_v) / (max_v - min_v)
        perceptual_diff *= 255
        perceptual_diff = perceptual_diff.cpu().numpy()
        perceptual_diff = Image.fromarray(perceptual_diff.astype(np.uint8))
        
        # finish transformation
        perceptual_diff_tensor = self.base_transforms_diss(perceptual_diff).unsqueeze(0).cuda()
        entropy_tensor = self.base_transforms_diss(entropy_img).unsqueeze(0).cuda()
        distance_tensor = self.base_transforms_diss(distance).unsqueeze(0).cuda()
        
        # hot encode semantic map
        semantic_tensor[semantic_tensor == 255] = 20  # 'ignore label is 20'
        semantic_tensor = one_hot_encoding(semantic_tensor, 20).unsqueeze(0).cuda()
        
        # run dissimilarity
        with torch.no_grad():
            if self.prior:
                diss_pred = F.softmax(
                    self.diss_model(image_tensor, syn_image_tensor, semantic_tensor, entropy_tensor,
                                    perceptual_diff_tensor,
                                    distance_tensor), dim=1)
            else:
                diss_pred = F.softmax(self.diss_model(image_tensor, syn_image_tensor, semantic_tensor), dim=1)
        diss_pred = diss_pred.cpu().numpy()
        
        # do ensemble if necessary
        if self.ensemble:
            diss_pred = diss_pred[:, 1, :, :] * 0.75 + entropy_tensor.cpu().numpy() * 0.25
        else:
            diss_pred = diss_pred[:, 1, :, :]
        diss_pred = np.array(Image.fromarray(diss_pred.squeeze()).resize((image_og_w, image_og_h)))
        
        out = {'anomaly_score': torch.tensor(diss_pred), 'segmentation': torch.tensor(seg_final)}
        
        return out['anomaly_score']

    def set_seeds(self, seed=0):
        # set seeds for reproducibility
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    def get_segmentation(self):
        # Get Segmentation Net
        assert_and_infer_cfg(self.opt, train_mode=False)
        self.opt.dataset_cls = cityscapes
        net = network.get_net(self.opt, criterion=None)
        net = torch.nn.DataParallel(net).cuda()
        print('Segmentation Net Built.')
        snapshot = os.path.join(os.getcwd(), os.path.dirname(__file__), self.opt.snapshot)
        self.seg_net, _ = restore_snapshot(net, optimizer=None, snapshot=snapshot,
                                           restore_optimizer_bool=False)
        self.seg_net.eval()
        print('Segmentation Net Restored.')
        
    def get_synthesis(self):
        # Get Synthesis Net
        print('Synthesis Net Built.')
        self.opt.checkpoints_dir = os.path.join(os.getcwd(), os.path.dirname(__file__), self.opt.checkpoints_dir)
        self.syn_net = Pix2PixModel(self.opt)
        self.syn_net.eval()
        print('Synthesis Net Restored')
        
    def get_dissimilarity(self, ours):
        # Get Dissimilarity Net
        if ours:
            config_diss = os.path.join(os.getcwd(), os.path.dirname(__file__), 'image_dissimilarity/configs/test/ours_configuration.yaml')
        else:
            config_diss = os.path.join(os.getcwd(), os.path.dirname(__file__), 'image_dissimilarity/configs/test/baseline_configuration.yaml')
    
        with open(config_diss, 'r') as stream:
            config_diss = yaml.load(stream, Loader=yaml.FullLoader)
    
        self.prior = config_diss['model']['prior']
        self.ensemble = config_diss['ensemble']
    
        if self.prior:
            self.diss_model = DissimNetPrior(**config_diss['model']).cuda()
        else:
            self.diss_model = DissimNet(**config_diss['model']).cuda()
    
        print('Dissimilarity Net Built.')
        save_folder = os.path.join(os.getcwd(), os.path.dirname(__file__), config_diss['save_folder'])
        model_path = os.path.join(save_folder,
                                  '%s_net_%s.pth' % (config_diss['which_epoch'], config_diss['experiment_name']))
        model_weights = torch.load(model_path)
        self.diss_model.load_state_dict(model_weights)
        self.diss_model.eval()
        print('Dissimilarity Net Restored')
        
    def get_transformations(self):
        # Transform images to Tensor based on ImageNet Mean and STD
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])
    
        # synthesis necessary pre-process
        self.transform_semantic = transforms.Compose(
            [transforms.Resize(size=(256, 512), interpolation=Image.NEAREST), transforms.ToTensor()])
        self.transform_image_syn = transforms.Compose(
            [transforms.Resize(size=(256, 512), interpolation=Image.BICUBIC), transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5),
                                  (0.5, 0.5, 0.5))])
    
        # dissimilarity pre-process
        self.vgg_diff = VGG19_difference().cuda()
        self.base_transforms_diss = transforms.Compose(
            [transforms.Resize(size=(256, 512), interpolation=Image.NEAREST), transforms.ToTensor()])
        self.norm_transform_diss = transforms.Compose(
            [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # imageNet normamlization
        self.to_pil = ToPILImage()
    
if __name__ == '__main__':
    import bdlb
    
    # define fishyscapes test parameters
    fs = bdlb.load(benchmark="fishyscapes")
    # automatically downloads the dataset
    data = fs.get_dataset('Static')
    detector = AnomalyDetector(True)
    metrics = fs.evaluate(detector.estimator_worker, data)
    
    print('My method achieved {:.2f}% AP'.format(100 * metrics['AP']))
    print('My method achieved {:.2f}% FPR@95TPR'.format(100 * metrics['FPR@95%TPR']))
    print('My method achieved {:.2f}% auroc'.format(100 * metrics['auroc']))
