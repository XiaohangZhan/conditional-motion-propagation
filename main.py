import argparse
import os
import yaml
import multiprocessing as mp

from utils import dist_init
from trainer import Trainer

def main(args):
    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in config.items():
        setattr(args, k, v)

    # exp path
    if not hasattr(args, 'exp_path'):
        args.exp_path = os.path.dirname(args.config)

    # dist init
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    dist_init(args.launcher, backend='nccl')

    # train
    trainer = Trainer(args)
    trainer.run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Kinematics')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--launcher', default='pytorch', type=str)
    parser.add_argument('--load-iter', default=None, type=int)
    parser.add_argument('--load-path', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()

    main(args)
