#!/bin/sh

#gcloud storage rsync -r -d predicted/ gs://omura-1/ ||\
gsutil -m rsync -r -d -u predicted/ gs://omura-1/predicted/

