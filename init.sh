#!/bin/sh

download_dataset () {
  desc=$1
  dataset=$2
  dataset_fn="$(basename -- $dataset)"

  echo "* Download the $desc dataset: $dataset"
  kaggle competitions download -c $dataset && \
  unzip -o ${dataset_fn}.zip -d data/$dataset_fn
  rm -f ${dataset_fn}.zip
}

# Official Dataset

#download_dataset "kaggle competitions" "child-mind-institute-detect-sleep-states"


download_kernel_output () {
  desc=$1
  dataset=$2
  dataset_fn="$(basename -- $dataset)"

  echo "* Download the $desc dataset: $dataset"
  kaggle kernels output $dataset -p data/$dataset_fn
}

download_kernel_output "Sleep-Critical-point-Prepare-Data" "werus23/sleep-critical-point-prepare-data"
