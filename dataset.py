import numpy as np
import io
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils.flowlib import read_flo_file
from utils import image_crop, image_resize, image_flow_crop, image_flow_resize, flow_sampler, image_flow_aug, flow_aug

class ColorAugmentation(object):

    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec == None:
            eig_vec = torch.Tensor([
                [ 0.4009,  0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [ 0.4203, -0.6948, -0.5836],
            ])
        if eig_val == None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(means=torch.zeros_like(self.eig_val))*0.1
        quatity = torch.mm(self.eig_val*alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor


def pil_loader(img_str, ch):
    buff = io.BytesIO(img_str)
    if ch == 1:
        return Image.open(buff)
    else:
        with Image.open(buff) as img:
            img = img.convert('RGB')
        return img


def pil_loader_str(img_str, ch):
    if ch == 1:
        return Image.open(img_str)
    else:
        with Image.open(img_str) as img:
            img = img.convert('RGB')
        return img


class ImageFlowDataset(Dataset):

    def __init__(self, meta_file, config, phase):
        self.img_transform = transforms.Compose([
            transforms.Normalize(config['data_mean'], config['data_div'])
        ])
        print("building dataset from {}".format(meta_file))
        self.flow_file_type = config['flow_file_type']
        self.metas = []
        self.num = 0
        for mf in meta_file:
            with open(mf, 'r') as f:
                lines = f.readlines()
            self.num += len(lines)
            for line in lines:
                if self.flow_file_type == "flo":
                    img0_path, img1_path, flow_path = line.rstrip().split()
                    self.metas.append((img0_path, img1_path, flow_path))
                elif self.flow_file_type == "jpg":
                    img0_path, img1_path, flow_path_x, flow_path_y = line.rstrip().split()
                    self.metas.append((img0_path, img1_path, flow_path_x, flow_path_y))
                else:
                    raise Exception("No such flow_file_type: {}".format(self.flow_file_type))
        print("read meta done, total: {}".format(self.num))

        self.phase = phase

        self.short_size = config.get('short_size', None)
        self.long_size = config.get('long_size', None)
        self.crop_size = config.get('crop_size', None)
        self.sample_strategy = config['sample_strategy']
        self.sample_bg_ratio = config['sample_bg_ratio']
        self.nms_ks = config['nms_ks']
        self.max_num_guide = config['max_num_guide']

        if self.phase == "train":
            self.aug_flip = config['image_flow_aug'].get('flip', False)
            self.aug_reverse = config['flow_aug'].get('reverse', False)
            self.aug_scale = config['flow_aug'].get('scale', False)
            self.aug_rotate = config['flow_aug'].get('rotate', False)
 
    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img1_fn = self.metas[idx][0]
        img2_fn = self.metas[idx][1]
        if self.flow_file_type == 'flo':
            flowname = self.metas[idx][2]
            flow = read_flo_file(flowname) # h, w, 2
        else:
            flownamex = self.metas[idx][2]
            flownamey = self.metas[idx][3]
            flowx = np.array(Image.open(flownamex)).astype(np.float32) / 255 * 100 - 50
            flowy = np.array(Image.open(flownamey)).astype(np.float32) / 255 * 100 - 50
            flow = np.concatenate((flowx[:,:,np.newaxis], flowy[:,:,np.newaxis]), axis=2)
        img1 = pil_loader_str(img1_fn, ch=3)
        img2 = pil_loader_str(img2_fn, ch=3)

        ## check size
        assert img1.height == flow.shape[0]
        assert img1.width == flow.shape[1]
        assert img2.height == flow.shape[0]
        assert img2.width == flow.shape[1]

        ## resize
        if self.short_size is not None or self.long_size is not None:
            img1, img2, flow, ratio = image_flow_resize(
                img1, img2, flow, short_size=self.short_size,
                long_size=self.long_size)

        ## crop
        if self.crop_size is not None:
            img1, img2, flow, offset = image_flow_crop(
                img1, img2, flow, self.crop_size, self.phase)

        ## augmentation
        if self.phase == 'train':
            # image flow aug
            img1, img2, flow = image_flow_aug(img1, img2, flow, flip_horizon=self.aug_flip)
            # flow aug
            flow = flow_aug(flow, reverse=self.aug_reverse,
                            scale=self.aug_scale, rotate=self.aug_rotate)

        ## transform
        img1 = torch.from_numpy(np.array(img1).astype(np.float32).transpose((2,0,1)))
        img2 = torch.from_numpy(np.array(img2).astype(np.float32).transpose((2,0,1)))
        img1 = self.img_transform(img1)
        img2 = self.img_transform(img2)

        ## sparse sampling
        sparse_flow, mask = flow_sampler(
            flow, strategy=self.sample_strategy,
            bg_ratio=self.sample_bg_ratio, nms_ks=self.nms_ks,
            max_num_guide=self.max_num_guide) # (h,w,2), (h,w,2)

        flow = torch.from_numpy(flow.transpose((2, 0, 1)))
        sparse_flow = torch.from_numpy(sparse_flow.transpose((2, 0, 1)))
        mask = torch.from_numpy(mask.transpose((2, 0, 1)).astype(np.float32))
        return img1, sparse_flow, mask, flow, img2


class ImageDataset(Dataset):

    def __init__(self, meta_file, config):
        self.img_transform = transforms.Compose([
            transforms.Normalize(config['data_mean'], config['data_div'])
        ])
        print("building dataset from {}".format(meta_file))
        with open(meta_file, 'r') as f:
            lines = f.readlines()
        self.num = len(lines)
        self.metas = [l.rstrip() for l in lines]
        print("read meta done, total: {}".format(self.num))

        self.short_size = config.get('short_size', None)
        self.long_size = config.get('long_size', None)
        self.crop_size = config.get('crop_size', None)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img_fn = self.metas[idx]
        img = pil_loader_str(img_fn, ch=3)

        ## resize
        if self.short_size is not None or self.long_size is not None:
            img, size = image_resize(img, short_size=self.short_size, long_size=self.long_size)

        ## crop
        if self.crop_size is not None:
            img, offset = image_crop(img, self.crop_size)

        ## transform
        img = torch.from_numpy(np.array(img).astype(np.float32).transpose((2,0,1)))
        img = self.img_transform(img)

        return img, torch.LongTensor([idx]), torch.LongTensor(offset), torch.LongTensor(size)
