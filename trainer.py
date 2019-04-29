import os
import time

import torch
import torch.optim
import torch.distributed as dist
import torchvision.utils as vutils
from torch.utils.data import DataLoader

import models
import utils


class Trainer(object):
    def __init__(self, args):

        # get rank
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        if self.rank == 0:
            # mkdir path
            if not os.path.exists('{}/events'.format(args.exp_path)):
                os.makedirs('{}/events'.format(args.exp_path))
            if not os.path.exists('{}/images'.format(args.exp_path)):
                os.makedirs('{}/images'.format(args.exp_path))
            if not os.path.exists('{}/logs'.format(args.exp_path)):
                os.makedirs('{}/logs'.format(args.exp_path))
            if not os.path.exists('{}/checkpoints'.format(args.exp_path)):
                os.makedirs('{}/checkpoints'.format(args.exp_path))
    
            # logger
            if args.trainer['tensorboard'] and not args.extract:
                try:
                    from tensorboardX import SummaryWriter
                except:
                    raise Exception("Please switch off \"tensorboard\" "
                                    "in your config file if you do not "
                                    "want to use it, otherwise install it.")
                self.tb_logger = SummaryWriter('{}/events'.format(args.exp_path))
            else:
                self.tb_logger = None
            if args.validate:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_offline_val.txt'.format(args.exp_path))
            elif args.extract:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_extract.txt'.format(args.exp_path))
            else:
                self.logger = utils.create_logger(
                    'global_logger',
                    '{}/logs/log_train.txt'.format(args.exp_path))
        
        # create model
        self.model = models.__dict__[args.model['arch']](args.model, dist_model=True)
    
        # optionally resume from a checkpoint
        assert not (args.load_iter is not None and args.load_path is not None)
        if args.load_iter is not None:
            self.model.load_state("{}/checkpoints".format(args.exp_path),
                                  args.load_iter, args.resume)
            self.start_iter = args.load_iter
        else:
            self.start_iter = 0
        if args.load_path is not None:
            self.model.load_pretrain(args.load_path)
        self.curr_step = self.start_iter

        # lr scheduler
        if not (args.validate or args.extract): # train
            self.lr_scheduler = utils.StepLRScheduler(
                self.model.optim, args.model['lr_steps'],
                args.model['lr_mults'], args.model['lr'], args.model['warmup_lr'],
                args.model['warmup_steps'], last_iter=self.start_iter-1)

        # Data loader
        if args.data['memcached']:
            from dataset_mc import McImageFlowDataset, McImageDataset
            imageflow_dataset = McImageFlowDataset
            image_dataset = McImageDataset
        else:
            from dataset import ImageFlowDataset, ImageDataset
            imageflow_dataset = ImageFlowDataset
            image_dataset = ImageDataset

        if not (args.validate or args.extract): # train
            train_dataset = imageflow_dataset(args.data['train_source'], args.data, 'train')
            train_sampler = utils.DistributedGivenIterationSampler(
                train_dataset, args.model['total_iter'],
                args.data['batch_size'], last_iter=self.start_iter-1)
            self.train_loader = DataLoader(
                train_dataset, batch_size=args.data['batch_size'], shuffle=False,
                num_workers=args.data['workers'], pin_memory=False, sampler=train_sampler)

        if not args.extract: # train or offline validation
            val_dataset = imageflow_dataset(args.data['val_source'], args.data, 'val')
            val_sampler = utils.DistributedSequentialSampler(val_dataset)
            self.val_loader = DataLoader(
                val_dataset, batch_size=args.data['batch_size_test'], shuffle=False,
                num_workers=args.data['workers'], pin_memory=False, sampler=val_sampler)
        else: # extract
            extract_dataset = image_dataset(args.extract_source, args.data)
            self.extract_metas = extract_dataset.metas
            extract_sampler = utils.DistributedSequentialSampler(extract_dataset)
            self.extract_loader = DataLoader(
                extract_dataset, batch_size=1, shuffle=False,
                num_workers=1, pin_memory=False, sampler=extract_sampler)

        self.args = args

    def run(self):

        # validate only
        if self.args.validate:
            self.validate('off_val')
            return

        # extract
        if self.args.extract:
            self.extract()
            return
    
        if self.args.trainer['initial_val']:
            self.validate('on_val')
    
        # train
        self.train()
    
    
    def train(self):
    
        btime_rec = utils.AverageMeter(10)
        dtime_rec = utils.AverageMeter(10)
        npts_rec = utils.AverageMeter(1000)
        recorder = {}
        for rec in self.args.trainer['loss_record']:
            recorder[rec] = utils.AverageMeter(10)
    
        self.model.switch_to('train')
    
        end = time.time()
        for i, (image, sparse, mask, flow_target, rgb_target) in enumerate(self.train_loader):
            self.curr_step = self.start_iter + i
            self.lr_scheduler.step(self.curr_step)
            curr_lr = self.lr_scheduler.get_lr()[0]

            # measure data loading time
            dtime_rec.update(time.time() - end)
            npts_rec.update(int(torch.sum(mask)/mask.size(0)/mask.size(1)))
    
            assert image.shape[0] > 0
            image = image.cuda()
            sparse = sparse.cuda()
            mask = mask.cuda()
            flow_target = flow_target.cuda()
            rgb_target = rgb_target.cuda()

            self.model.set_input(image, torch.cat([sparse, mask], dim=1),
                                 flow_target, rgb_target)
            loss_dict = self.model.step()
            for k in loss_dict.keys():
                recorder[k].update(utils.reduce_tensors(loss_dict[k]).item()) 

            btime_rec.update(time.time() - end)
            end = time.time()
    
            # logging
            if self.rank == 0 and self.curr_step % self.args.trainer['print_freq'] == 0:
                loss_str = ""
                if self.tb_logger is not None:
                    self.tb_logger.add_scalar('npts', npts_rec.avg, self.curr_step)
                    self.tb_logger.add_scalar('lr', curr_lr, self.curr_step)
                for k in recorder.keys():
                    if self.tb_logger is not None:
                        self.tb_logger.add_scalar('train_{}'.format(k), recorder[k].avg,
                                                  self.curr_step + 1)
                    loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})\t'.format(
                        k, loss=recorder[k])

                self.logger.info(
                    'Iter: [{0}/{1}]\t'.format(self.curr_step, len(self.train_loader)) +
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                        batch_time=btime_rec) +
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                        data_time=dtime_rec) +
                    loss_str +
                    'NPts {num_pts.val} ({num_pts.avg:.1f})\t'.format(num_pts=npts_rec) +
                    'lr {lr:.2g}'.format(lr=curr_lr))
    
            # validate
            if ((self.curr_step + 1) % self.args.trainer['val_freq'] == 0 or
                (self.curr_step + 1) == self.args.model['total_iter']):
                self.validate('on_val')
    
            # save
            if (self.rank == 0 and
                ((self.curr_step + 1) % self.args.trainer['save_freq'] == 0 or
                 (self.curr_step + 1) == self.args.model['total_iter'])):
                self.model.save_state("{}/checkpoints".format(self.args.exp_path),
                                      self.curr_step + 1)


    def validate(self, phase):
        btime_rec = utils.AverageMeter(0)
        dtime_rec = utils.AverageMeter(0)
        npts_rec = utils.AverageMeter(0)
        recorder = {}
        for rec in self.args.trainer['loss_record']:
            recorder[rec] = utils.AverageMeter(10)
   
        self.model.switch_to('eval')
    
        end = time.time()
        all_together = []
        for i, (image, sparse, mask, flow_target, rgb_target) in enumerate(self.val_loader):
            if ('val_iter' in self.args.trainer and
                self.args.trainer['val_iter'] != -1 and
                i == self.args.trainer['val_iter']):
                break
    
            assert image.shape[0] > 0

            dtime_rec.update(time.time() - end)
            npts_rec.update(int(torch.sum(mask) / mask.size(0) / mask.size(1)))
    
            image = image.cuda()
            sparse = sparse.cuda()
            mask = mask.cuda()
            flow_target = flow_target.cuda()
            rgb_target = rgb_target.cuda()

            self.model.set_input(image, torch.cat([sparse, mask], dim=1),
                                 flow_target, rgb_target)
            tensor_dict, loss_dict = self.model.eval()
            for k in loss_dict.keys():
                recorder[k].update(utils.reduce_tensors(loss_dict[k]).item()) 
            btime_rec.update(time.time() - end)
            end = time.time()

            # tb visualize
            if self.rank == 0:
                if (i >= self.args.trainer['val_disp_start_iter'] and
                    i < self.args.trainer['val_disp_end_iter']):
                    all_together.append(utils.visualize_tensor(
                        image, mask, tensor_dict['flow_tensors'],
                        tensor_dict['common_tensors'], tensor_dict['rgb_tensors'],
                        self.args.data['data_mean'], self.args.data['data_div']))
                if (i == self.args.trainer['val_disp_end_iter'] and
                    self.args.trainer['val_disp_end_iter'] >
                        self.args.trainer['val_disp_start_iter']):
                    all_together = torch.cat(all_together, dim=2)
                    grid = vutils.make_grid(
                        all_together, nrow=1, normalize=True,
                        range=(0, 255), scale_each=False)
                    if self.tb_logger is not None:
                        self.tb_logger.add_image('Image_' + phase, grid, self.curr_step + 1)

        # logging
        if self.rank == 0:
            loss_str = ""
            for k in recorder.keys():
                if self.tb_logger is not None:
                    self.tb_logger.add_scalar('val_{}'.format(k),
                                              recorder[k].avg, self.curr_step + 1)
                loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})\t'.format(
                    k, loss=recorder[k])

            self.logger.info(
                'Validation Iter: [{0}]\t'.format(self.curr_step) +
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    batch_time=btime_rec) +
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                    data_time=dtime_rec) +
                loss_str +
                'NPts {num_pts.val} ({num_pts.avg:.1f})\t'.format(num_pts=npts_rec))

        self.model.switch_to("train")

    def extract(self):
        raise NotImplemented
