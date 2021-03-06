#!/bin/bash

# set -ex
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
dir_path=$(dirname "${BASH_SOURCE[0]}")

# python "${dir_path}/$1"
# ex)
# bash main.sh src/data/loader.py == python "${dir_path}/src/data/loader.py"

project_name="${1:-SCCNN_LSTM_Hybrid}"
model_name="${2:-SCCNN_LSTM_Hybrid}"
gpu="${3:-0}"
for i in $(seq 1 20); do
    python run.py \
        log.project_name="$project_name" \
        log.model_name="$model_name" \
        optimizer.lr=1e-4 \
        log.device.gpu="$gpu" \
        data.roi=$i
done

