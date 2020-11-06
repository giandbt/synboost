#!/bin/bash
#CUDA_VISIBLE_DEVICES=14 python train.py --config configs/train/baseline_configuration.yaml --gpu_ids 0 --seed 0
#CUDA_VISIBLE_DEVICES=14 python train.py --config configs/train/baseline_configuration.yaml --gpu_ids 0 --seed 1
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train/default_configuration.yaml --gpu_ids 0 --seed 3
#CUDA_VISIBLE_DEVICES=15 python train.py --config configs/train/default_configuration.yaml --gpu_ids 0 --seed 3
#CUDA_VISIBLE_DEVICES=14 python train.py --config configs/train/baseline_configuration.yaml --gpu_ids 0 --seed 4