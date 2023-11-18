#!/bin/sh

python3 -m build
rm dist/*.tar.gz
kaggle datasets version -p dist/ -m "" --dir-mode skip

python3 -m pip wheel . -w dist/wheels
kaggle datasets version -p dist/wheels -m ""

