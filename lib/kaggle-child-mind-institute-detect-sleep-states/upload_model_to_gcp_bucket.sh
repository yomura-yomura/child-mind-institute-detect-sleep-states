#!/bin/sh

model_dir_path=$1

if [ $# -eq 0 ]; then
  echo "error: model_dir_path argument must be given"
  return 1
fi

gcloud storage cp -r --preserve-symlinks $model_dir_path gs://omura-1/ ||\
gsutil cp -n -r $model_dir_path gs://omura-1/
