#!/bin/sh

exp_name=$1

if [ $# -eq 0 ]; then
  echo "error: exp_name argument must be given"
  return 1
fi

if [ -e "output/train/$exp_name" ]; then
  echo "error: $exp_name already exists"
  return 1
fi

gcloud storage cp -r --preserve-symlinks gs://omura-1/$exp_name/ output/train/
