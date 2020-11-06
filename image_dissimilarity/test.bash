#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/test/road_anomaly_configuration.yaml --gpu_ids 0
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/test/fs_static_configuration.yaml --gpu_ids 0
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/test/fs_lost_found_configuration.yaml --gpu_ids 0