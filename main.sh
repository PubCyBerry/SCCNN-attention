#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
dir_path=$(dirname "${BASH_SOURCE[0]}")

# python "${dir_path}/src/data/loader.py"
python "${dir_path}/$1"
