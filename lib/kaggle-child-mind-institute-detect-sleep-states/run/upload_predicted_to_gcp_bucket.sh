#!/bin/sh

#gcloud storage rsync -r -d predicted/ gs://omura-1/ ||\
gsutil -m rsync -r -u  -x "(?!.*/train/.*)" predicted/ gs://omura-1/predicted/

