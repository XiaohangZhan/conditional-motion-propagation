import torch
import yaml
import argparse
import sys
from packaging import version

sys.path.append('.')
import models
import os

import pdb

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Kinematics')
parser.add_argument('--config', required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

def main():
    exp_dir = os.path.dirname(args.config)
    
    with open(args.config) as f:
        if version.parse(yaml.version >= "5.1"):
            config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            config = yaml.load(f)

    for k, v in config.items():
        setattr(args, k, v)
    
    model = models.modules.__dict__[args.model['module']['arch']](args.model['module'])
    model = torch.nn.DataParallel(model)
    
    ckpt_path = exp_dir + '/checkpoints/ckpt_iter_{}.pth.tar'.format(args.iter)
    save_path = exp_dir + '/checkpoints/convert_iter_{}.pth.tar'.format(args.iter)
    ckpt = torch.load(ckpt_path)
    weight = ckpt['state_dict']
    model.load_state_dict(weight, strict=True)
    model = model.module.image_encoder
    
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()
