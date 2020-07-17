import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, ToTensor
import yaml
from options.test_options import TestOptions
import sys

sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(__file__), 'image_segmentation'))
import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg
from image_synthesis.models.pix2pix_model import Pix2PixModel
from image_dissimilarity.models.dissimilarity_model import DissimNetPrior
from image_dissimilarity.models.vgg_features import VGG19_difference
from image_dissimilarity.data.cityscapes_dataset import one_hot_encoding

# Common options for all models
TestOptions = TestOptions()
opt = TestOptions.parse()
torch.cuda.empty_cache()

# Get Segmentation Net
assert_and_infer_cfg(opt, train_mode=False)
opt.dataset_cls = cityscapes
net = network.get_net(opt, criterion=None)
net = torch.nn.DataParallel(net).cuda()
print('Segmentation Net built.')
snapshot = os.path.join(os.getcwd(), os.path.dirname(__file__), opt.snapshot)
seg_net, _ = restore_snapshot(net, optimizer=None, snapshot=opt.snapshot, restore_optimizer_bool=False)
seg_net.eval()
print('Segmentation Net Restored.')

# Get Synthesis Net
world_size = 1
rank = 0
print('Synthesis Net built.')
opt.checkpoints_dir = os.path.join(os.getcwd(), os.path.dirname(__file__), opt.checkpoints_dir)
syn_net = Pix2PixModel(opt)
syn_net.eval()
print('Synthesis Net Restored')

# Get Dissimilarity Net
config_diss = os.path.join(os.getcwd(), os.path.dirname(__file__), 'image_dissimilarity/configs/test/ours_configuration.yaml')
with open(config_diss, 'r') as stream:
    config_diss = yaml.load(stream, Loader=yaml.FullLoader)

if config_diss['model']['prior']:
    diss_model = DissimNetPrior(**config_diss['model']).cuda()
    print('Dissimilarity Net built.')
    
    model_path = os.path.join(config_diss['save_folder'],
                              '%s_net_%s.pth' % (config_diss['which_epoch'], config_diss['experiment_name']))
    model_weights = torch.load(model_path)
    diss_model.load_state_dict(model_weights)
    diss_model.eval()
    print('Dissimilarity Net Restored')

# Get RGB Original Images
data_dir = opt.demo_folder
images = os.listdir(data_dir)
if len(images) == 0:
    print('There are no images at directory %s. Check the data path.' % (data_dir))
else:
    print('There are %d images to be processed.' % (len(images)))
images.sort()

# Transform images to Tensor based on ImageNet Mean and STD
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])

# Create save directory
if not os.path.exists(opt.results_dir):
    os.makedirs(opt.results_dir)

# synthesis necessary pre-process
transform_semantic = transforms.Compose(
    [transforms.Resize(size=(256, 512), interpolation=Image.NEAREST), transforms.ToTensor()])
transform_image_syn = transforms.Compose(
    [transforms.Resize(size=(256, 512), interpolation=Image.BICUBIC), transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5),
                          (0.5, 0.5, 0.5))])

# dissimilarity pre-process
vgg_diff = VGG19_difference().cuda()
base_transforms_diss = transforms.Compose(
    [transforms.Resize(size=(256, 512), interpolation=Image.NEAREST), transforms.ToTensor()])
norm_transform_diss = transforms.Compose(
    [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # imageNet normamlization
to_pil = ToPILImage()

# Loop around all figures
with torch.no_grad():
    for img_id, img_name in enumerate(images):
        print('Predicting image %i out of %i' % (img_id + 1, len(images)))
        img_dir = os.path.join(data_dir, img_name)
        img = Image.open(img_dir).convert('RGB').resize((2048, 1024))
        img_tensor = img_transform(img)
        
        # predict segmentation
        img_final = img_tensor.unsqueeze(0).cuda()
        seg_outs = seg_net(img_final)
        
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
        for label_id, train_id in opt.dataset_cls.id_to_trainid.items():
            label_out[np.where(seg_final == train_id)] = label_id
        label_img = Image.fromarray((label_out).astype(np.uint8))
        
        # prepare for synthesis
        label_tensor = transform_semantic(label_img) * 255.0
        label_tensor[label_tensor == 255] = 35  # 'unknown' is opt.label_nc
        image_tensor = transform_image_syn(img)
        # Get instance map in right format. Since prediction doesn't have instance map, we use semantic instead
        instance_tensor = label_tensor.clone()
        
        # run synthesis
        syn_input = {'label': label_tensor.unsqueeze(0), 'instance': instance_tensor.unsqueeze(0),
                     'image': image_tensor.unsqueeze(0)}
        
        generated = syn_net(syn_input, mode='inference')
        
        image_numpy = (np.transpose(generated.squeeze().cpu().numpy(), (1, 2, 0)) + 1) / 2.0
        synthesis_final_img = Image.fromarray((image_numpy * 255).astype(np.uint8))
        
        # prepare dissimilarity
        entropy = entropy.cpu().numpy()
        distance = distance.cpu().numpy()
        entropy_img = Image.fromarray(entropy.astype(np.uint8).squeeze())
        distance = Image.fromarray(distance.astype(np.uint8).squeeze())
        semantic = Image.fromarray((seg_final).astype(np.uint8))
        
        # get initial transformation
        semantic_tensor = base_transforms_diss(semantic) * 255
        syn_image_tensor = base_transforms_diss(synthesis_final_img)
        image_tensor = base_transforms_diss(img)
        syn_image_tensor = norm_transform_diss(syn_image_tensor).unsqueeze(0).cuda()
        image_tensor = norm_transform_diss(image_tensor).unsqueeze(0).cuda()
        
        # get softmax difference
        perceptual_diff = vgg_diff(image_tensor, syn_image_tensor)
        
        min_v = torch.min(perceptual_diff.squeeze())
        max_v = torch.max(perceptual_diff.squeeze())
        perceptual_diff = (perceptual_diff.squeeze() - min_v) / (max_v - min_v)
        perceptual_diff *= 255
        perceptual_diff = perceptual_diff.cpu().numpy()
        perceptual_diff = Image.fromarray(perceptual_diff.astype(np.uint8))
        
        # finish transformation
        perceptual_diff_tensor = base_transforms_diss(perceptual_diff).unsqueeze(0).cuda()
        entropy_tensor = base_transforms_diss(entropy_img).unsqueeze(0).cuda()
        distance_tensor = base_transforms_diss(distance).unsqueeze(0).cuda()
        
        # hot encode semantic map
        semantic_tensor[semantic_tensor == 255] = 20  # 'ignore label is 20'
        semantic_tensor = one_hot_encoding(semantic_tensor, 20).unsqueeze(0).cuda()
        
        # run dissimilarity
        diss_pred = diss_model(image_tensor, syn_image_tensor, semantic_tensor, entropy_tensor, perceptual_diff_tensor,
                           distance_tensor)
        
        diss_pred = F.softmax(diss_pred, dim=1)
        
        diss_pred = diss_pred.cpu().numpy()
        diss_pred = diss_pred[:, 1, :, :] * 0.75 + entropy_tensor.cpu().numpy() * 0.25
        
        # Save image
        diss_pred = (diss_pred * 255).astype(np.uint8)
        diss_pred = Image.fromarray(diss_pred.squeeze())
        diss_pred.resize((1024, 512)).save(os.path.join(opt.results_dir, os.path.basename(img_name)))

