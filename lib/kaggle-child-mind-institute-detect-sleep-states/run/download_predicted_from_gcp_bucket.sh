#!/bin/sh

gsutil -m rsync -r -d -u gs://omura-1/predicted/ predicted/
