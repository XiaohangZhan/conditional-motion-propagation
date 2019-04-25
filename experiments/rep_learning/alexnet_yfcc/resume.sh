#!/bin/bash
work_path=$(dirname $0)
python -u main.py \
    --config $work_path/config.yaml --launcher pytorch \
    --load-iter 10000 \
    --resume
