#!/bin/bash

set -ex
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
dir_path=$(dirname "${BASH_SOURCE[0]}")

# python "${dir_path}/$1"
# ex)
# bash main.sh src/data/loader.py == python "${dir_path}/src/data/loader.py"

for i in $(seq 0 20); do
    python run.py log.project_name='SCCNN_LSTM_roi_rank' log.model_name='SCCNN_LSTM_roi_rank' optimizer.lr=1e-4 log.device.gpu=0 data.roi=$i
done

