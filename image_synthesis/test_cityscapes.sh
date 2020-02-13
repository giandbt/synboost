#!/bin/bash
python test.py \
    --batchSize 1 \
    --netG condconv \
    --checkpoints_dir /home/giancarlo/Documents/thesis/CC-FPSE/checkpoints \
    --which_epoch 200 \
    --name cs_pretrained \
    --dataset_mode cityscapes \
    --dataroot /media/giancarlo/Samsung_T5/data/cityscapes \
    --use_vae