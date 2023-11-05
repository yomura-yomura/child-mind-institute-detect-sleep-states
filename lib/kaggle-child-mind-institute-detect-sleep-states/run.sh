#!/bin/sh

python3 run/train.py exp_name=exp004-fold0 split=fold_0
python3 run/train.py exp_name=exp004-fold1 split=fold_1
python3 run/train.py exp_name=exp004-fold2 split=fold_2
python3 run/train.py exp_name=exp004-fold3 split=fold_3
python3 run/train.py exp_name=exp004-fold4 split=fold_4
