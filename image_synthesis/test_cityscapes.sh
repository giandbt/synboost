#!/bin/bash
python test.py \
    --batchSize 1 \
    --netG condconv \
    --checkpoints_dir checkpoints \
    --which_epoch latest \
    --name cityscapes_own_cc_fpse \
    --dataset_mode cityscapes \
    --dataroot datasets/cityscapes \
    --use_vae