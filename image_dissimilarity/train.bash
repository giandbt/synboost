#!/bin/bash
CUDA_VISIBLE_DEVICES=6 python train.py --config configs/train/default_configuration.yaml --gpu_ids 0
#CUDA_VISIBLE_DEVICES=14 python train.py --config configs/train/epfl_configuration.yaml --gpu_ids 0
#CUDA_VISIBLE_DEVICES=14 python train.py --config configs/train/batch_norm_configuration.yaml --gpu_ids 0
#CUDA_VISIBLE_DEVICES=14 python train.py --config configs/train/spade_configuration.yaml --gpu_ids 0
#CUDA_VISIBLE_DEVICES=14 python train.py --config configs/train/guided_configuration.yaml --gpu_ids 0
#CUDA_VISIBLE_DEVICES=14 python train.py --config configs/train/no_correlation_configuration.yaml --gpu_ids 0
#CUDA_VISIBLE_DEVICES=14 python train.py --config configs/train/no_semantic_configuration.yaml --gpu_ids 0
#CUDA_VISIBLE_DEVICES=14 python train.py --config configs/train/no_pretrained_configuration.yaml --gpu_ids 0
