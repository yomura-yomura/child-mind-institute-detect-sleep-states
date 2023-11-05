#!/bin/sh

#EXP_NAME=exp004
EXP_NAME=exp005-lstm-feature
ARGS=feature_extractor=LSTMFeatureExtractor

python3 run/train.py exp_name=${EXP_NAME}-fold0 split=fold_0 $ARGS
python3 run/train.py exp_name=${EXP_NAME}-fold1 split=fold_1 $ARGS
python3 run/train.py exp_name=${EXP_NAME}-fold2 split=fold_2 $ARGS
python3 run/train.py exp_name=${EXP_NAME}-fold3 split=fold_3 $ARGS
python3 run/train.py exp_name=${EXP_NAME}-fold4 split=fold_4 $ARGS
