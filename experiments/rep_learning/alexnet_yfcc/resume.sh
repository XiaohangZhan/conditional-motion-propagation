#!/bin/bash
work_path=$(dirname $0)
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p VI_SP_VA_1080TI -x SH-IDC1-10-5-36-37 -n8 --gres=gpu:8 --ntasks-per-node=8 \
python -u main.py \
    --config $work_path/config.yaml \
    --load-iter 42000 \
    --load-cmp-only \
    --resume
