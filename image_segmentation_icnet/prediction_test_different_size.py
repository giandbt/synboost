import os
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from shutil import copyfile
import datetime


import libs.models as models


N_CLASS = 19
color_list = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
color_map = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
             (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
             (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]
up_kwargs = {'mode': 'bilinear', 'align_corners': True}


def transform(img):
    img = cv2.imread(img)
    IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
    img = img - IMG_MEAN
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).cuda()
    return img

def transform_rgb(img):
    img = cv2.imread(img, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32)

    img /= 255
    IMG_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
    IMG_VARS = np.array((0.229, 0.224, 0.225), dtype=np.float32)

    img -= IMG_MEAN
    img /= IMG_VARS

    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).cuda()
    return img



def makeTestlist(dir,start=0,end=1525):
    out = []
    floder = os.listdir(dir)
    for f in floder:
        floder_dir = os.path.join(dir, f)
        for i in os.listdir(floder_dir):
            out.append(os.path.join(floder_dir, i))
    out.sort()
    return out[start:end]


def WholeTest(args, model, size=1.0):
    
    save_dir = args.output_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    entropy_fdr = os.path.join(save_dir, 'entropy_icnet')
    distance_fdr = os.path.join(save_dir, 'logit_distance_icnet')
    semantic_fdr = os.path.join(save_dir, 'semantic_icnet')
    semantic_label_fdr = os.path.join(save_dir, 'semantic_label_ids_icnet')
    
    if not os.path.exists(entropy_fdr):
        os.makedirs(entropy_fdr)
    
    if not os.path.exists(distance_fdr):
        os.makedirs(distance_fdr)
    
    if not os.path.exists(semantic_fdr):
        os.makedirs(semantic_fdr)
    
    if not os.path.exists(semantic_label_fdr):
        os.makedirs(semantic_label_fdr)
    
    # creates temporary folder to adapt format to image synthesis
    if not os.path.exists(os.path.join(save_dir, 'temp')):
        os.makedirs(os.path.join(save_dir, 'temp'))
        os.makedirs(os.path.join(save_dir, 'temp', 'gtFine', 'val'))
        os.makedirs(os.path.join(save_dir, 'temp', 'leftImg8bit', 'val'))
        
    net = model.cuda()
    net.eval()
    saved_state_dict = torch.load(args.resume)
    net.load_state_dict(saved_state_dict)
    print('model loaded')
    img_list = [os.path.join(args.input_dir, image) for image in os.listdir(args.input_dir) if '.jpg' or '.png' in image]
    for i in img_list:
        if os.path.isdir(i):
            continue
        name = i
        image_save_path = os.path.join(save_dir, 'temp', 'leftImg8bit', 'val', os.path.basename(name)[:-4] + '_leftImg8bit' + os.path.basename(name)[-4:])
        copyfile(name, image_save_path)
        
        with torch.no_grad():
            if args.rgb:
                img = transform_rgb(i)
            else:
                img = transform(i)
            _, _, origin_h, origin_w = img.size()
            h, w = int(origin_h*size), int(origin_w*size)
            img = F.upsample(img, size=(h, w), mode="bilinear", align_corners=True)
            out = net(img)[0]
            out = F.upsample(out, size=(origin_h, origin_w), mode='bilinear', align_corners=True)
            output = F.softmax(out, dim=1)
            
            # get entropy
            entropy = torch.sum(-output * torch.log(output), dim=1)
            entropy = (entropy - entropy.min()) / entropy.max()

            # get logit distance
            distance, _ = torch.topk(output, 2, dim=1)
            max_logit = distance[:, 0, :, :]
            max2nd_logit = distance[:, 1, :, :]
            map_logit = max_logit - max2nd_logit
            distance = 1 - (map_logit - map_logit.min()) / map_logit.max()
            
            result = out.argmax(dim=1)[0]
            result = result.data.cpu().squeeze().numpy()
            entropy = entropy.data.cpu().squeeze().numpy()
            distance = distance.data.cpu().squeeze().numpy()
            
            row, col = result.shape
            dst = np.ones((row, col), dtype=np.uint8) * 255
            for i in range(19):
                dst[result == i] = color_list[i]
            print(name, " done!")
            

            save_name_entropy = os.path.join(entropy_fdr, os.path.basename(name)[:-4] + '.png')
            save_name_distance = os.path.join(distance_fdr, os.path.basename(name)[:-4] + '.png')
            save_name_semantic = os.path.join(semantic_fdr, os.path.basename(name)[:-4] + '.png')
            save_name_semantic_label = os.path.join(semantic_label_fdr, os.path.basename(name)[:-4] + '.png')

            cv2.imwrite(save_name_entropy, entropy*255)
            cv2.imwrite(save_name_distance, distance*255)
            cv2.imwrite(save_name_semantic, result)
            cv2.imwrite(save_name_semantic_label, dst)

            

            cv2.imwrite(os.path.join(save_dir, 'temp', 'gtFine', 'val', os.path.basename(name)[:-4] + '_instanceIds.png'),
                        dst)
            cv2.imwrite(os.path.join(save_dir, 'temp', 'gtFine', 'val', os.path.basename(name)[:-4] + '_labelIds.png'),
                        dst)
            
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch \
                Segmentation Crop Prediction')
    parser.add_argument('--input_dir', type=str,
                        default="/home/lxt/data/Cityscapes/leftImg8bit/test",
                        help='training dataset folder (default: \
                              $(HOME)/data)')
    parser.add_argument("--input_disp_dir", type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="/home/giancarlo/Desktop/temp2",
                        help='output directory of the model_test, for saving the seg_models')
    parser.add_argument("--resume", type=str, default="/home/lxt/Desktop/Seg_model_ZOO/CNL_net_4w_ohem/CS_scenes_40000.pth")
    parser.add_argument("--start",type=int,default=0,help="start index of crop test")
    parser.add_argument("--end",type=int,default=5000,help="end index of crop test")
    parser.add_argument("--gpu",type=str,default="0",help="which gpu to use")
    parser.add_argument("--arch",type=str,default=None, help="which network are used")
    parser.add_argument("--size",type=float,default=1.0,help="ratio of the input images")
    parser.add_argument("--rgb",type=int,default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    test_list =[os.path.join(args.input_dir, image) for image in os.listdir(args.input_dir) if '.jpg' or '.png' in image]
    model= models.__dict__[args.arch](num_classes=19, data_set="cityscapes")
    WholeTest(args, model=model, size=args.size)