#!/bin/sh

gsutil -m rsync -r -u -x "(?!.*/train/.*)" gs://omura-1/predicted/ predicted/
