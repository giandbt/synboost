"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict
import shutil

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html


opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b])])
    visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()
        
synthesis_fdr = os.path.join(opt.results_dir, 'synthesis_spade')
# Cleans output folders and deletes temporary files
if not os.path.exists(synthesis_fdr):
    os.makedirs(synthesis_fdr)

source_fdr = os.path.join(opt.results_dir, 'image-synthesis_spade/test_latest/images/synthesized_image')
for image_name in os.listdir(source_fdr):
    shutil.move(os.path.join(source_fdr, image_name), os.path.join(synthesis_fdr, image_name))

shutil.rmtree(os.path.join(opt.results_dir, 'image-synthesis_spade'))
shutil.rmtree(os.path.join(opt.results_dir, 'temp'))
