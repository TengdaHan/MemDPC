import os
import sys
import time
import re
import argparse
import numpy as np
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
plt.switch_backend('agg')

sys.path.append('../')
from dataset import K400Dataset, UCF101Dataset
from model import MemDPC_BD

import utils.augmentation as A
from utils.utils import AverageMeter, save_checkpoint, Logger,\
calc_topk_accuracy, neq_load_customized, MultiStepLR_Restart_Multiplier

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='resnet18', type=str)
    parser.add_argument('--model', default='memdpc', type=str)
    parser.add_argument('--dataset', default='ucf101', type=str)
    parser.add_argument('--seq_len', default=5, type=int, help='number of frames in each video block')
    parser.add_argument('--num_seq', default=8, type=int, help='number of video blocks')
    parser.add_argument('--pred_step', default=3, type=int)
    parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
    parser.add_argument('--mem_size', default=1024, type=int, help='memory size')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--gpu', default='0,1', type=str)
    parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
    parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
    parser.add_argument('--img_dim', default=128, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('-j', '--workers', default=16, type=int)
    args = parser.parse_args()
    return args


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    device = torch.device('cuda')
    num_gpu = len(str(args.gpu).split(','))
    args.batch_size = num_gpu * args.batch_size

    ### model ###
    if args.model == 'memdpc':
        model = MemDPC_BD(sample_size=args.img_dim, 
                        num_seq=args.num_seq, 
                        seq_len=args.seq_len, 
                        network=args.net, 
                        pred_step=args.pred_step,
                        mem_size=args.mem_size)
    else: 
        raise NotImplementedError('wrong model!')

    model.to(device)
    model = nn.DataParallel(model)
    model_without_dp = model.module

    ### optimizer ###
    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    ### data ###
    transform = transforms.Compose([
        A.RandomSizedCrop(size=224, consistent=True, p=1.0), # crop from 256 to 224
        A.Scale(size=(args.img_dim,args.img_dim)),
        A.RandomHorizontalFlip(consistent=True),
        A.RandomGray(consistent=False, p=0.25),
        A.ColorJitter(0.5, 0.5, 0.5, 0.25, consistent=False, p=1.0),
        A.ToTensor(),
        A.Normalize()
    ])

    train_loader = get_data(transform, 'train')
    val_loader = get_data(transform, 'val')

    if 'ucf' in args.dataset: 
        lr_milestones_eps = [300,400]
    elif 'k400' in args.dataset: 
        lr_milestones_eps = [120,160]
    else: 
        lr_milestones_eps = [1000] # NEVER
    lr_milestones = [len(train_loader) * m for m in lr_milestones_eps]
    print('=> Use lr_scheduler: %s eps == %s iters' % (str(lr_milestones_eps), str(lr_milestones)))
    lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=lr_milestones, repeat=1)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_acc = 0
    args.iteration = 1

    ### restart training ###
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resumed checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            args.iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            model_without_dp.load_state_dict(checkpoint['state_dict'])
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print('[WARNING] Not loading optimizer states')
            print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.resume))
            sys.exit(0)

    # logging tools
    args.img_path, args.model_path = set_path(args)
    args.logger = Logger(path=args.img_path)
    args.logger.log('args=\n\t\t'+'\n\t\t'.join(['%s:%s'%(str(k),str(v)) for k,v in vars(args).items()]))

    args.writer_val = SummaryWriter(logdir=os.path.join(args.img_path, 'val'))
    args.writer_train = SummaryWriter(logdir=os.path.join(args.img_path, 'train'))
    
    torch.backends.cudnn.benchmark = True

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)

        train_loss, train_acc = train_one_epoch(train_loader, 
                                                model, 
                                                criterion, 
                                                optimizer, 
                                                lr_scheduler, 
                                                device, 
                                                epoch, 
                                                args)
        val_loss, val_acc = validate(val_loader, 
                                     model, 
                                     criterion, 
                                     device, 
                                     epoch, 
                                     args)

        # save check_point
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_dict = {'epoch': epoch,
                     'state_dict': model_without_dp.state_dict(),
                     'best_acc': best_acc,
                     'optimizer': optimizer.state_dict(),
                     'iteration': args.iteration}
        save_checkpoint(save_dict, is_best, 
            filename=os.path.join(args.model_path, 'epoch%s.pth.tar' % str(epoch)), 
            keep_all=False)

    print('Training from ep %d to ep %d finished' 
        % (args.start_epoch, args.epochs))
    sys.exit(0)


def train_one_epoch(data_loader, model, criterion, optimizer, lr_scheduler, device, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = [[AverageMeter(), AverageMeter()], # forward top1, top5
                [AverageMeter(), AverageMeter()]] # backward top1, top5
    
    model.train()
    end = time.time()
    tic = time.time()

    for idx, input_seq in enumerate(data_loader):
        data_time.update(time.time() - end)
        
        input_seq = input_seq.to(device)
        B = input_seq.size(0)
        loss, loss_step, acc, extra = model(input_seq)

        for i in range(2):
            top1, top5 = acc[i].mean(0) # average acc across multi-gpus
            accuracy[i][0].update(top1.item(), B)
            accuracy[i][1].update(top5.item(), B)

        loss = loss.mean() # average loss across multi-gpus
        losses.update(loss.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.6f}\t'
                  'Acc: {acc[0][0].val:.4f}\t'
                  'T-data:{dt.val:.2f} T-batch:{bt.val:.2f}\t'.format(
                   epoch, idx, len(data_loader),
                   loss=losses, acc=accuracy, dt=data_time, bt=batch_time))

            args.writer_train.add_scalar('local/loss',   losses.val,         args.iteration)
            args.writer_train.add_scalar('local/F-top1', accuracy[0][0].val, args.iteration)
            args.writer_train.add_scalar('local/F-top5', accuracy[0][1].val, args.iteration)
            args.writer_train.add_scalar('local/B-top1', accuracy[1][0].val, args.iteration)
            args.writer_train.add_scalar('local/B-top5', accuracy[1][1].val, args.iteration)

        args.iteration += 1
        if lr_scheduler is not None: lr_scheduler.step()

    print('Epoch: [{0}]\t'
          'T-epoch:{t:.2f}\t'.format(epoch, t=time.time()-tic))

    args.writer_train.add_scalar('global/loss',   losses.avg,         epoch)
    args.writer_train.add_scalar('global/F-top1', accuracy[0][0].avg, epoch)
    args.writer_train.add_scalar('global/F-top5', accuracy[0][1].avg, epoch)
    args.writer_train.add_scalar('global/B-top1', accuracy[1][0].avg, epoch)
    args.writer_train.add_scalar('global/B-top5', accuracy[1][1].avg, epoch)

    return losses.avg, np.mean([accuracy[0][0].avg, accuracy[1][0].avg])


def validate(data_loader, model, criterion, device, epoch, args):
    losses = AverageMeter()
    accuracy = [[AverageMeter(), AverageMeter()], # forward top1, top5
                [AverageMeter(), AverageMeter()]] # backward top1, top5
    
    model.eval()

    with torch.no_grad():
        for idx, input_seq in enumerate(data_loader):            
            input_seq = input_seq.to(device)
            B = input_seq.size(0)
            loss, loss_step, acc, extra = model(input_seq)

            for i in range(2):
                top1, top5 = acc[i].mean(0) # average acc across multi-gpus
                accuracy[i][0].update(top1.item(), B)
                accuracy[i][1].update(top5.item(), B)

            loss = loss.mean() # average loss across multi-gpus
            losses.update(loss.item(), B)

    print('Epoch: [{0}/{1}]\t'
          'Loss {loss.val:.6f}\t'
          'Acc: {acc[0][0].val:.4f}\t'.format(
           epoch, args.epochs,
           loss=losses, acc=accuracy))

    args.writer_val.add_scalar('global/loss',   losses.avg,         epoch)
    args.writer_val.add_scalar('global/F-top1', accuracy[0][0].avg, epoch)
    args.writer_val.add_scalar('global/F-top5', accuracy[0][1].avg, epoch)
    args.writer_val.add_scalar('global/B-top1', accuracy[1][0].avg, epoch)
    args.writer_val.add_scalar('global/B-top5', accuracy[1][1].avg, epoch)

    return losses.avg, np.mean([accuracy[0][0].avg, accuracy[1][0].avg])


def get_data(transform, mode='train'):
    print('Loading {} dataset for {}'.format(args.dataset, mode))
    if args.dataset == 'k400':
        dataset = K400Dataset(mode=mode,
                              transform=transform,
                              seq_len=args.seq_len,
                              num_seq=args.num_seq,
                              downsample=args.ds)
    elif args.dataset == 'ucf101':
        dataset = UCF101Dataset(mode=mode,
                         transform=transform,
                         seq_len=args.seq_len,
                         num_seq=args.num_seq,
                         downsample=args.ds)
    else:
        raise NotImplementedError('dataset not supported')

    sampler = data.RandomSampler(dataset)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  sampler=sampler,
                                  shuffle=False,
                                  num_workers=args.workers,
                                  pin_memory=True,
                                  drop_last=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader

def set_path(args):
    if args.resume: 
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_{args.prefix}/{args.model}_{args.dataset}-{args.img_dim}_{args.net}_\
mem{args.mem_size}_bs{args.batch_size}_lr{args.lr}_seq{args.num_seq}_pred{args.pred_step}_\
len{args.seq_len}_ds{args.ds}'.format(args=args)

    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): 
        os.makedirs(img_path)
    if not os.path.exists(model_path): 
        os.makedirs(model_path)

    return img_path, model_path

if __name__ == '__main__':
    args = parse_args()
    main(args)
