#!/bin/bash

set -e
set -x

if [ ! -d "dvclive" ]; then
  echo "Please run this script from repository root" >&2
  exit 1
fi

python setup.py sdist
python setup.py bdist_wheel --universal