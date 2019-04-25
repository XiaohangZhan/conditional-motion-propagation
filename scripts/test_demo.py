import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
import numpy as np
import yaml
import sys
import time
from scipy import interpolate
import os
sys.path.append("../")
#sys.path.append("/mnt/lustre/share/zhanxiaohang/videoseg/lib/pydensecrf")
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import flowlib
import models
import utils

exp = '../experiments/vip+mpii/resnet50_skiplayer_residual'
load_iter = 46000
config = "{}/config.yaml".format(exp)


class ArgObj(object):
    def __init__(self):
        pass

def image_resize(img, short_size):
    w, h = img.width, img.height
    if w < h:
        neww = short_size
        newh = int(short_size / float(w) * h)
    else:
        neww = int(short_size / float(h) * w)
        newh = short_size
    img = img.resize((neww, newh), Image.BICUBIC)
    return img

def image_crop(img, crop_size):
    pad_h = max(crop_size[0] - img.height, 0)
    pad_w = max(crop_size[1] - img.width, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        border = (pad_w_half, pad_h_half, pad_w - pad_w_half, pad_h - pad_h_half)
        img = ImageOps.expand(img, border=border, fill=(0,0,0))
    hoff = (img.height - crop_size[0]) // 2
    woff = (img.width - crop_size[1]) // 2
    return img.crop((woff, hoff, woff+crop_size[1], hoff+crop_size[0]))

def flow_crop(flow, crop_size):
    pad_h = max(crop_size[0] - img.height, 0)
    pad_w = max(crop_size[1] - img.width, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        flow_expand = np.zeros((img.height + pad_h, img.width + pad_w, 2), dtype=np.float32)
        flow_expand[pad_h_half:pad_h_half+img.height, pad_w_half:pad_w_half+img.width, :] = flow
        flow = flow_expand
    hoff = (img.height - crop_size[0]) // 2
    woff = (img.width - crop_size[1]) // 2
    return flow[hoff:hoff+crop_size[0], woff:woff+crop_size[1], :]

def image_flow_warp(img, flow, mask_th=1, copy=True, interp=True, interp_mode=0):
    
    warp_img = np.zeros(img.shape, dtype=img.dtype)
    flow_mask = (np.abs(flow[:,:,0]) > mask_th) | (np.abs(flow[:,:,1]) > mask_th)
    pts = np.where(flow_mask)
    vx_pts = flow[:,:,0][pts].astype(np.int)
    vy_pts = flow[:,:,1][pts].astype(np.int)
    v = flow[:,:,0][pts] ** 2 + flow[:,:,1][pts] ** 2
    sortidx = np.argsort(v)
    warp_pts = (pts[0] + vy_pts, pts[1] + vx_pts)
    warp_pts = (np.clip(warp_pts[0], 0, img.shape[0]-1), np.clip(warp_pts[1], 0, img.shape[1]-1))
    warp_pts = (warp_pts[0][sortidx], warp_pts[1][sortidx])
    pts = [pts[0][sortidx], pts[1][sortidx]]
    for c in range(3):
        if copy:
            warp_img[:,:,c][~flow_mask] = img[:,:,c][~flow_mask]
        warp_img[:,:,c][warp_pts] = img[:,:,c][pts]
    if interp:
        holes = ((warp_img.sum(axis=2) == 0) & flow_mask)
        if interp_mode == 0:
            hpts = np.where(holes)
            opts = (hpts[0]-flow[:,:,1][hpts].astype(np.int), hpts[1]-flow[:,:,0][hpts].astype(np.int))
            opts = (np.clip(opts[0], 0, img.shape[0]-1), np.clip(opts[1], 0, img.shape[1]-1))
            for c in range(3):
                warp_img[:,:,c][hpts] = img[:,:,c][opts]
        else:
            for c in range(3):
                warp_img[:,:,c][holes] = interpolate.griddata(np.where(~holes), warp_img[:,:,c][~holes], np.where(holes), method='cubic')
    return warp_img
    
    
class Demo(object):
    def __init__(self, configfn, load_iter):
        args = ArgObj()
        with open(configfn) as f:
            config = yaml.load(f)
        for k, v in config.items():
            setattr(args, k, v)
        setattr(args, 'load_iter', load_iter)
        setattr(args, 'exp_path', os.path.dirname(configfn))
        
        self.model = models.__dict__[args.model['arch']](args.model, dist_model=False)
        
        self.model.load_state("{}/checkpoints".format(args.exp_path), args.load_iter, False)
        self.model.switch_to('eval')
      
        self.data_mean = args.data['data_mean']
        self.data_div = args.data['data_div']
        
        self.img_transform = transforms.Compose([
            transforms.Normalize(self.data_mean, self.data_div)])

        self.args = args
        
    def def_input(self, image, repeat=1):
        self.rgb = image
        tensor = self.img_transform(torch.from_numpy(np.array(image).astype(np.float32).transpose((2,0,1))))
        self.image = tensor.unsqueeze(0).repeat(repeat,1,1,1)
        
    def run(self, arrows):
        sparse = np.zeros((1, 2, self.image.size(2), self.image.size(3)), dtype=np.float32)
        mask = np.zeros((1, 2, self.image.size(2), self.image.size(3)), dtype=np.float32)
        for arr in arrows:
            sparse[0, :, int(arr[1]), int(arr[0])] = np.array(arr[2:4])
            mask[0, :, int(arr[1]), int(arr[0])] = np.array([1, 1])
        image = self.image.cuda()
        sparse = torch.from_numpy(sparse).cuda()
        mask = torch.from_numpy(mask).cuda()

        self.model.set_input(image, torch.cat([sparse, mask], dim=1))
        self.model.forward_gie()
        out_flow = self.model.flow.detach().cpu().numpy()[0].transpose((1,2,0))
        out_warped = torch.clamp(utils.unormalize(self.model.warped.detach().cpu(), mean=self.data_mean, div=self.data_div), 0, 255).numpy()[0].transpose((1,2,0))
        out_rgb_gen = torch.clamp(utils.unormalize(self.model.rgb_gen.detach().cpu(), mean=self.data_mean, div=self.data_div), 0, 255).numpy()[0].transpose((1,2,0))
        return out_flow, out_warped, out_rgb_gen


if __name__ == "__main__":
    demo = Demo(config, load_iter)

    fn = '/mnt/lustre/share/panxingang/VIP/group1/videos9/000000000801/000000000798.jpg'
    img = Image.open(fn).convert("RGB")
    img = image_resize(img, demo.args.data['short_size'])
    img = image_crop(img, demo.args.data['crop_size'])
    demo.def_input(img)

    coords = [[80.64993548387096, 126.88270967741937, -12.155870967741976, -16.71432258064516]]
    flow, warped, rgb_gen = demo.run(coords)
