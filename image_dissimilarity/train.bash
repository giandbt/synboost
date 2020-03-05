#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train/hot_encoded_configuration.yaml --gpu_ids 0
