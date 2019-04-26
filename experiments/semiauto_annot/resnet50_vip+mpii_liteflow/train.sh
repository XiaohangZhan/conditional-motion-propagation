#!/bin/bash
work_path=$(dirname $0)
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p AD -n8 --gres=gpu:8 --ntasks-per-node=8 \
python -u main.py \
    --config $work_path/config.yaml --launcher slurm
