#!/bin/sh

#EXP_NAME=exp004
#EXP_NAME=exp005-lstm-feature-2
#EXP_NAME=exp005-lstm-feature-3
#EXP_NAME=exp006-lstm-feature-fp16
#EXP_NAME=exp009-lstm-feature-half-lr
#EXP_NAME=exp010-lstm-feature-mlp-decoder

#EXP_NAME=exp011-lstm-feature-1d-fp16
#EXP_NAME=exp012-lstm-feature
#ARGS="batch_size=8 optimizer.lr=0.0001 feature_extractor=LSTMFeatureExtractor model.encoder_name=resnet18 epoch=100"

#EXP_NAME=exp013-stacked-gru-feature
#ARGS="batch_size=8 optimizer.lr=0.0001 feature_extractor=StackedGRUFeatureExtractor feature_extractor.num_layers=3 model.encoder_name=resnet18 epoch=100"

#EXP_NAME=exp011-lstm-feature-1d-fp16-2
#ARGS="model_dim=1 model=Spec1D batch_size=8 optimizer.lr=0.0001 feature_extractor=LSTMFeatureExtractor model.encoder_name=resnet34 epoch=100"

EXP_NAME=exp014-lstm-feature
ARGS="feature_extractor=LSTMFeatureExtractor model.encoder_name=resnet18"

echo $ARGS

python3 run/train.py --multirun exp_name=${EXP_NAME} split=fold_0,fold_1,fold_2,fold_3,fold_4 $ARGS
