import os
from PIL import Image
import numpy as np
import cv2
from collections import OrderedDict
import shutil
import torch
from torch.backends import cudnn
import torch.onnx
import torchvision.transforms as transforms
import onnx

from options.test_options import TestOptions


import sys
sys.path.insert(0, './image_segmentation')
import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

TestOptions = TestOptions()
opt = TestOptions.parse()

def convert_segmentation_model(model_name = 'segmentation.onnx'):


    assert_and_infer_cfg(opt, train_mode=False)
    cudnn.benchmark = False
    torch.cuda.empty_cache()

    # Get segmentation Net
    opt.dataset_cls = cityscapes
    net = network.get_net(opt, criterion=None)
    net = torch.nn.DataParallel(net).cuda()
    print('Segmentation Net built.')
    net, _ = restore_snapshot(net, optimizer=None, snapshot=opt.snapshot, restore_optimizer_bool=False)
    net.eval()
    print('Segmentation Net Restored.')

    # Input to the model
    batch_size = 1

    x = torch.randn(batch_size, 3, 512, 1024, requires_grad=True).cuda()
    torch_out = net(x)

    # Export the model
    torch.onnx.export(net.module,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      model_name,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                    'output' : {0 : 'batch_size'}})


def convert_synthesis_model(model_name = 'synthesis.onnx'):

    import sys
    sys.path.insert(0, './image_synthesis')
    from models.pix2pix_model import Pix2PixModel

    world_size = 1
    rank = 0

    # Corrects where dataset is in necesary format
    opt.dataroot = os.path.join(opt.results_dir, 'temp')

    opt.world_size = world_size
    opt.gpu = 0
    opt.mpdist = False

    model = Pix2PixModel(opt)
    model.eval()

    # Input to the model
    batch_size = 1

    label = torch.randn(batch_size, 1, 256, 512, requires_grad=True).cuda()
    instance = torch.randn(batch_size, 1, 256, 512, requires_grad=True).cuda()
    image = torch.randn(batch_size, 3, 256, 512, requires_grad=True).cuda()
    path = ['dummy']
    data = {'label': label, 'instance': instance, 'image': image, 'path': path}
    import pdb; pdb.set_trace()
    torch_out = model(data, mode='inference')

    # Export the model
    torch.onnx.export(model,               # model being run
                      (data, 'inference'),                         # model input (or a tuple for multiple inputs)
                      model_name,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                    'output' : {0 : 'batch_size'}})
if __name__ == '__main__':
    #convert_segmentation_model()
    convert_synthesis_model()