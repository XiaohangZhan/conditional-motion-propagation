#!/bin/bash
work_path=$(dirname $0)
launcher=$1
if [ "$launcher" == "pytorch" ]; then
    python -u main.py \
        --config $work_path/config.yaml --launcher pytorch
elif [ "$launcher" == "slurm" ]; then
    partition=$2
    GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $partition -n8 \
        --gres=gpu:8 --ntasks-per-node=8 \
        python -u main.py \
            --config $work_path/config.yaml --launcher slurm
else
    echo "No such launcher: $launcher"
fi
