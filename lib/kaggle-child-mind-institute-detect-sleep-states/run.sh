#!/bin/sh

#EXP_NAME=exp004
#EXP_NAME=exp005-lstm-feature-2
#EXP_NAME=exp005-lstm-feature-3
#EXP_NAME=exp006-lstm-feature-fp16
#EXP_NAME=exp009-lstm-feature-half-lr
#EXP_NAME=exp010-lstm-feature-mlp-decoder

#EXP_NAME=exp011-lstm-feature-1d-fp16
EXP_NAME=exp012-lstm-feature

ARGS="batch_size=8 optimizer.lr=0.0001 feature_extractor=LSTMFeatureExtractor model.encoder_name=resnet18 epoch=100"

echo $ARGS

python3 run/train.py exp_name=${EXP_NAME} split=fold_0 $ARGS
python3 run/train.py exp_name=${EXP_NAME} split=fold_1 $ARGS
python3 run/train.py exp_name=${EXP_NAME} split=fold_2 $ARGS
python3 run/train.py exp_name=${EXP_NAME} split=fold_3 $ARGS
python3 run/train.py exp_name=${EXP_NAME} split=fold_4 $ARGS
