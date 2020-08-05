#!/bin/bash
#CUDA_VISIBLE_DEVICES=14 python test_multiple.py --config configs/test/road_anomaly_configuration.yaml --gpu_ids 0 --num_exp 3
#CUDA_VISIBLE_DEVICES=14 python test_multiple.py --config configs/test/fs_static_configuration.yaml --gpu_ids 0 --num_exp 3
#CUDA_VISIBLE_DEVICES=14 python test_multiple.py --config configs/test/fs_lost_found_configuration.yaml --gpu_ids 0 --num_exp 3

CUDA_VISIBLE_DEVICES=15 python test.py --config configs/test/road_anomaly_configuration.yaml --gpu_ids 0
CUDA_VISIBLE_DEVICES=15 python test.py --config configs/test/fs_static_configuration.yaml --gpu_ids 0
CUDA_VISIBLE_DEVICES=15 python test.py --config configs/test/fs_lost_found_configuration.yaml --gpu_ids 0


#CUDA_VISIBLE_DEVICES=15 python test_ensemble.py --config configs/test/fs_static_configuration.yaml --gpu_ids 0
#CUDA_VISIBLE_DEVICES=15 python test_ensemble.py --config configs/test/fs_lost_found_configuration.yaml --gpu_ids 0
