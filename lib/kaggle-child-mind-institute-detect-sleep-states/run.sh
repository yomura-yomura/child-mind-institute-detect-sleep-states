#!/bin/sh

#EXP_NAME=exp004
#EXP_NAME=exp005-lstm-feature-2
#EXP_NAME=exp005-lstm-feature-3
#EXP_NAME=exp006-lstm-feature-fp16
#EXP_NAME=exp009-lstm-feature-half-lr
#EXP_NAME=exp010-lstm-feature-mlp-decoder

EXP_NAME=exp011-lstm-feature-1d-fp16
CONFIG_DIR=config/omura/v100/1d/

ARGS="batch_size=8 optimizer.lr=0.0001 --config-dir $CONFIG_DIR"

echo $ARGS

python3 run/train.py exp_name=${EXP_NAME}-fold0 split=fold_0 $ARGS
python3 run/train.py exp_name=${EXP_NAME}-fold1 split=fold_1 $ARGS
python3 run/train.py exp_name=${EXP_NAME}-fold2 split=fold_2 $ARGS
python3 run/train.py exp_name=${EXP_NAME}-fold3 split=fold_3 $ARGS
python3 run/train.py exp_name=${EXP_NAME}-fold4 split=fold_4 $ARGS
