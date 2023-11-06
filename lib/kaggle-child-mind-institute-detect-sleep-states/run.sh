#!/bin/sh

#EXP_NAME=exp004
#EXP_NAME=exp005-lstm-feature
#EXP_NAME=exp006-lstm-feature-fp16
#EXP_NAME=exp007-lstm-feature-1d-fp16
EXP_NAME=exp009-lstm-feature-half-lr

ARGS="feature_extractor=LSTMFeatureExtractor"
#ARGS="feature_extractor=LSTMFeatureExtractor use_amp=true model=Spec1D"

echo $ARGS

python3 run/train.py exp_name=${EXP_NAME}-fold0 split=fold_0 $ARGS
python3 run/train.py exp_name=${EXP_NAME}-fold1 split=fold_1 $ARGS
python3 run/train.py exp_name=${EXP_NAME}-fold2 split=fold_2 $ARGS
python3 run/train.py exp_name=${EXP_NAME}-fold3 split=fold_3 $ARGS
python3 run/train.py exp_name=${EXP_NAME}-fold4 split=fold_4 $ARGS
