#!/bin/bash
CUDA_VISIBLE_DEVICES=13 python test.py --config configs/test/road_anomaly_configuration.yaml --gpu_ids 0
#CUDA_VISIBLE_DEVICES=14 python test.py --config configs/test/lost_found_test_configuration.yaml --gpu_ids 0
