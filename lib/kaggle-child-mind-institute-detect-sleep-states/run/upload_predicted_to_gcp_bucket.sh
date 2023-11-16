#!/bin/sh

#gcloud storage rsync -r -d predicted/ gs://omura-1/ ||\
gsutil rsync -r -d -u -i predicted/ gs://omura-1/predicted/
