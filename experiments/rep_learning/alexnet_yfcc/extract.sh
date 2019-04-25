#!/bin/bash
work_path=$(dirname $0)
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p VITemp -n8 --gres=gpu:8 --ntasks-per-node=8 \
python -u main_km.py --config $work_path/config.yaml \
    --load-path $work_path/checkpoints/ckpt_iter_35000.pth.tar \
    --extract
