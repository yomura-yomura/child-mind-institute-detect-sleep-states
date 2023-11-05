#!/bin/sh
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run/train.py exp_name=manet-fold0 split=fold_0
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run/train.py exp_name=manet-fold1 split=fold_1
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run/train.py exp_name=manet-fold2 split=fold_2
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run/train.py exp_name=manet-fold3 split=fold_3
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run/train.py exp_name=manet-fold4 split=fold_4
