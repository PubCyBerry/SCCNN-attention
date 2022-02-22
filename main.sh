#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
dir_path=$(dirname "${BASH_SOURCE[0]}")

# python "${dir_path}/src/data/loader.py"
# python "${dir_path}/$1"

for var in {1}; do
    python run.py optimizer.lr=2e-5 log.device.gpu=0
    python run.py optimizer.lr=1e-5 log.device.gpu=0
    python run.py optimizer.lr=9e-6 log.device.gpu=0
    python run.py optimizer.lr=8e-6 log.device.gpu=0
    python run.py optimizer.lr=7e-6 log.device.gpu=0
    python run.py optimizer.lr=6e-6 log.device.gpu=0
    python run.py optimizer.lr=5e-6 log.device.gpu=0
    python run.py optimizer.lr=4e-6 log.device.gpu=0
    python run.py optimizer.lr=3e-6 log.device.gpu=0
    python run.py optimizer.lr=2e-6 log.device.gpu=0
    python run.py optimizer.lr=1e-6 log.device.gpu=0
done
