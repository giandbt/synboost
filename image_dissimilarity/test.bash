#!/bin/bash
CUDA_VISIBLE_DEVICES=7 python test.py --config configs/test/road_anomaly_configuration.yaml --gpu_ids 0
