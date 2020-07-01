import yaml
import torch
from torchvision.transforms import ToPILImage, ToTensor
import torchvision
import os

import sys
sys.path.append("./image_dissimilarity")
from util import trainer_util, metrics

def mae_features(config_file_path, gpu_ids, dataroot, data_origin):
    
    soft_fdr = os.path.join(dataroot, 'mae_features_' + data_origin)
    
    if not os.path.exists(soft_fdr):
        os.makedirs(soft_fdr)

    # load experiment setting
    with open(config_file_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    
    # activate GPUs
    config['gpu_ids'] = gpu_ids
    gpu = int(gpu_ids)
    
    # get data_loaders
    cfg_test_loader = config['test_dataloader']
    cfg_test_loader['dataset_args']['dataroot'] = dataroot
    test_loader = trainer_util.get_dataloader(cfg_test_loader['dataset_args'], cfg_test_loader['dataloader_args'])
    
    class VGG19(torch.nn.Module):
        def __init__(self, requires_grad=False):
            super().__init__()
            vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
    
            self.slice1 = torch.nn.Sequential()
            self.slice2 = torch.nn.Sequential()
            self.slice3 = torch.nn.Sequential()
            self.slice4 = torch.nn.Sequential()
            self.slice5 = torch.nn.Sequential()
            for x in range(2):
                self.slice1.add_module(str(x), vgg_pretrained_features[x])
            for x in range(2, 7):
                self.slice2.add_module(str(x), vgg_pretrained_features[x])
            for x in range(7, 12):
                self.slice3.add_module(str(x), vgg_pretrained_features[x])
            for x in range(12, 21):
                self.slice4.add_module(str(x), vgg_pretrained_features[x])
            for x in range(21, 30):
                self.slice5.add_module(str(x), vgg_pretrained_features[x])
            if not requires_grad:
                for param in self.parameters():
                    param.requires_grad = False
    
        def forward(self, X):
            h_relu1 = self.slice1(X)
            h_relu2 = self.slice2(h_relu1)
            h_relu3 = self.slice3(h_relu2)
            h_relu4 = self.slice4(h_relu3)
            h_relu5 = self.slice5(h_relu4)
            out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
            return out
        
    from  torch.nn.modules.upsampling import Upsample
    up5 = Upsample(scale_factor=16, mode='bicubic')
    up4 = Upsample(scale_factor=8, mode='bicubic')
    up3 = Upsample(scale_factor=4, mode='bicubic')
    up2 = Upsample(scale_factor=2, mode='bicubic')
    up1 = Upsample(scale_factor=1, mode='bicubic')
    to_pil = ToPILImage()
    
    # Going through visualization loader
    weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
    vgg = VGG19().cuda(gpu)
    
    with torch.no_grad():
        for i, data_i in enumerate(test_loader):
            print('Generating image %i out of %i'%(i+1, len(test_loader)))
            img_name = os.path.basename(data_i['original_path'][0])
            original = data_i['original'].cuda(gpu)
            synthesis = data_i['synthesis'].cuda(gpu)
            
            x_vgg, y_vgg = vgg(original), vgg(synthesis)
            feat5 = torch.mean(torch.abs(x_vgg[4] - y_vgg[4]), dim=1).unsqueeze(1)
            feat4 = torch.mean(torch.abs(x_vgg[3] - y_vgg[3]), dim=1).unsqueeze(1)
            feat3 = torch.mean(torch.abs(x_vgg[2] - y_vgg[2]), dim=1).unsqueeze(1)
            feat2 = torch.mean(torch.abs(x_vgg[1] - y_vgg[1]), dim=1).unsqueeze(1)
            feat1 = torch.mean(torch.abs(x_vgg[0] - y_vgg[0]), dim=1).unsqueeze(1)
            
            img_5 = up5(feat5)
            img_4 = up4(feat4)
            img_3 = up3(feat3)
            img_2 = up2(feat2)
            img_1 = up1(feat1)
            
            combined = weights[0] * img_1 + weights[1] * img_2 + weights[2] * img_3 + weights[3] * img_4 + weights[
                4] * img_5
            min_v = torch.min(combined.squeeze())
            max_v = torch.max(combined.squeeze())
            combined = (combined.squeeze() - min_v) / (max_v - min_v)
    
            combined = to_pil(combined.cpu())
            pred_name = 'mea_' + img_name
            combined.save(os.path.join(soft_fdr, pred_name))
            
            
        
if __name__ == '__main__':
    # input parameters
    config_file_path = 'image_dissimilarity/configs/visualization/default_configuration.yaml'
    gpu_ids = '6'
    dataroot = '/home/giancarlo/data/innosuisse/custom_both'
    data_origin = 'spade'
    mae_features(config_file_path, gpu_ids, dataroot, data_origin)
    


