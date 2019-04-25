import torch
import yaml
import argparse
import models
import os

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
        config = yaml.load(f)
    
    for k, v in config['common'].items():
        setattr(args, k, v)
    
    model_params = args.model[args.model['arch']]
    model = models.modules.__dict__[args.model['arch']](model_params)
    model.cuda()
    model = torch.nn.DataParallel(model)
    
    ckpt_path = exp_dir + '/checkpoints/ckpt_iter_{}.pth.tar'.format(args.iter)
    save_path = exp_dir + '/checkpoints/convert_iter_{}.pth.tar'.format(args.iter)
    ckpt = torch.load(ckpt_path)
    weight = ckpt['state_dict']
    model.load_state_dict(weight, strict=False)
    model.cpu()
    model = model.module.image_encoder
    
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()
