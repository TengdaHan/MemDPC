import os
import sys
import time
import argparse
import re
import numpy as np
import random
import json
from tqdm import tqdm
from tensorboardX import SummaryWriter

sys.path.append('../')
sys.path.append('../memdpc/')
from dataset import UCF101Dataset, HMDB51Dataset
from model_lc import LC
import utils.augmentation as A 
from utils.utils import AverageMeter, ConfusionMeter, save_checkpoint, \
calc_topk_accuracy, denorm, calc_accuracy, neq_load_customized, Logger

import torch
import torch.optim as optim
from torch.utils import data
import torch.nn as nn  
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='resnet18', type=str)
    parser.add_argument('--model', default='lc', type=str)
    parser.add_argument('--dataset', default='ucf101', type=str)
    parser.add_argument('--split', default=1, type=int)
    parser.add_argument('--seq_len', default=5, type=int)
    parser.add_argument('--num_seq', default=8, type=int)
    parser.add_argument('--num_class', default=101, type=int)
    parser.add_argument('--dropout', default=0.9, type=float)
    parser.add_argument('--ds', default=3, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--schedule', default=[], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--wd', default=1e-3, type=float, help='weight decay')
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--pretrain', default='random', type=str)
    parser.add_argument('--test', default='', type=str)
    parser.add_argument('--center_crop', action='store_true')
    parser.add_argument('--five_crop', action='store_true')
    parser.add_argument('--ten_crop', action='store_true')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--gpu', default='0,1', type=str)
    parser.add_argument('--print_freq', default=5, type=int)
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
    parser.add_argument('--train_what', default='last', type=str, help='Train what parameters?')
    parser.add_argument('--prefix', default='tmp', type=str)
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

    if args.dataset == 'ucf101': args.num_class = 101
    elif args.dataset == 'hmdb51': args.num_class = 51 

    ### classifier model ###
    if args.model == 'lc':
        model = LC(sample_size=args.img_dim, 
                   num_seq=args.num_seq, 
                   seq_len=args.seq_len, 
                   network=args.net,
                   num_class=args.num_class,
                   dropout=args.dropout,
                   train_what=args.train_what)
    else:
        raise ValueError('wrong model!')

    model.to(device)
    model = nn.DataParallel(model)
    model_without_dp = model.module 
    criterion = nn.CrossEntropyLoss()
    
    ### optimizer ### 
    params = None
    if args.train_what == 'ft':
        print('=> finetune backbone with smaller lr')
        params = []
        for name, param in model.module.named_parameters():
            if ('resnet' in name) or ('rnn' in name):
                params.append({'params': param, 'lr': args.lr/10})
            else:
                params.append({'params': param})
    elif args.train_what == 'last':
        print('=> train only last layer')
        params = []
        for name, param in model.named_parameters():
            if ('bone' in name) or ('agg' in name) or ('mb' in name) or ('network_pred' in name):
                param.requires_grad = False
            else: params.append({'params': param})
    else: 
        pass # train all layers
    
    print('\n===========Check Grad============')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print('=================================\n')

    if params is None: params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    
    ### scheduler ### 
    if args.dataset == 'hmdb51':
        step =  args.schedule
        if step == []: step = [150,250]
        lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=step, repeat=1)
    elif args.dataset == 'ucf101':
        step =  args.schedule
        if step == []: step = [300, 400]
        lr_lambda = lambda ep: MultiStepLR_Restart_Multiplier(ep, gamma=0.1, step=step, repeat=1)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    print('=> Using scheduler at {} epochs'.format(step))

    args.old_lr = None
    best_acc = 0
    args.iteration = 1

    ### if in test mode ###
    if args.test:
        if os.path.isfile(args.test):
            print("=> loading test checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test, map_location=torch.device('cpu'))
            try: 
                model_without_dp.load_state_dict(checkpoint['state_dict'])
            except:
                print('=> [Warning]: weight structure is not equal to test model; Load anyway ==')
                model_without_dp = neq_load_customized(model_without_dp, checkpoint['state_dict'])
            epoch = checkpoint['epoch']
            print("=> loaded testing checkpoint '{}' (epoch {})".format(args.test, checkpoint['epoch']))
        elif args.test == 'random':
            epoch = 0
            print("=> loaded random weights")
        else: 
            print("=> no checkpoint found at '{}'".format(args.test))
            sys.exit(0)

        args.logger = Logger(path=os.path.dirname(args.test))
        _, test_dataset = get_data(None, 'test')
        test_loss, test_acc = test(test_dataset, model, criterion, device, epoch, args)
        sys.exit()

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
            print("=> no checkpoint found at '{}'".format(args.resume))
            sys.exit(0)

    if (not args.resume) and args.pretrain:
        if args.pretrain == 'random':
            print('=> using random weights')
        elif os.path.isfile(args.pretrain):
            print("=> loading pretrained checkpoint '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain, map_location=torch.device('cpu'))
            model_without_dp = neq_load_customized(model_without_dp, checkpoint['state_dict'])
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(args.pretrain, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrain))
            sys.exit(0)

    ### data ###
    transform = transforms.Compose([
        A.RandomSizedCrop(consistent=True, size=224, p=1.0),
        A.Scale(size=(args.img_dim,args.img_dim)),
        A.RandomHorizontalFlip(consistent=True),
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.3, consistent=True),
        A.ToTensor(),
        A.Normalize()
    ])
    val_transform = transforms.Compose([
        A.RandomSizedCrop(consistent=True, size=224, p=0.3),
        A.Scale(size=(args.img_dim,args.img_dim)),
        A.RandomHorizontalFlip(consistent=True),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3, consistent=True),
        A.ToTensor(),
        A.Normalize()
    ])

    train_loader, _ = get_data(transform, 'train')
    val_loader, _ = get_data(val_transform, 'val')

    # setup tools
    args.img_path, args.model_path = set_path(args)
    args.writer_val = SummaryWriter(logdir=os.path.join(args.img_path, 'val'))
    args.writer_train = SummaryWriter(logdir=os.path.join(args.img_path, 'train'))
    torch.backends.cudnn.benchmark = True

    ### main loop ###
    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(train_loader, 
                                                model, 
                                                criterion, 
                                                optimizer, 
                                                device, 
                                                epoch, 
                                                args)
        val_loss, val_acc = validate(val_loader, 
                                     model, 
                                     criterion, 
                                     device, 
                                     epoch, 
                                     args)
        lr_scheduler.step(epoch)

        # save check_point
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_dict = {
            'epoch': epoch,
            'backbone': args.net,
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


def train_one_epoch(data_loader, model, criterion, optimizer, device, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    if args.train_what == 'last':
        model.eval()
        model.module.final_bn.train()
        model.module.final_fc.train()
        print('[Warning] train model with eval mode, except final layer')
    else:
        model.train()
    
    end = time.time()
    tic = time.time()

    for idx, (input_seq, target) in enumerate(data_loader):
        data_time.update(time.time() - end)
        input_seq = input_seq.to(device)
        target = target.to(device)
        B = input_seq.size(0)
        output, _ = model(input_seq)

        [_, N, D] = output.size()
        output = output.view(B*N, D)
        target = target.repeat(1, N).view(-1)

        loss = criterion(output, target)
        acc = calc_accuracy(output, target)

        losses.update(loss.item(), B)
        accuracy.update(acc.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.local_avg:.4f})\t'
                  'Acc: {acc.val:.4f} ({acc.local_avg:.4f})\t'
                  'T-data:{dt.val:.2f} T-batch:{bt.val:.2f}\t'.format(
                   epoch, idx, len(data_loader),
                   loss=losses, acc=accuracy, dt=data_time, bt=batch_time))
     
            args.writer_train.add_scalar('local/loss', losses.val, args.iteration)
            args.writer_train.add_scalar('local/accuracy', accuracy.val, args.iteration)

        args.iteration += 1
    print('Epoch: [{0}]\t'
          'T-epoch:{t:.2f}\t'.format(epoch, t=time.time()-tic))

    args.writer_train.add_scalar('global/loss', losses.avg, epoch)
    args.writer_train.add_scalar('global/accuracy', accuracy.avg, epoch)

    return losses.avg, accuracy.avg


def validate(data_loader, model, criterion, device, epoch, args):
    losses = AverageMeter()
    accuracy = AverageMeter()
    model.eval()
    with torch.no_grad():
        for idx, (input_seq, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_seq = input_seq.to(device)
            target = target.to(device)
            B = input_seq.size(0)
            output, _ = model(input_seq)

            [_, N, D] = output.size()
            output = output.view(B*N, D)
            target = target.repeat(1, N).view(-1)

            loss = criterion(output, target)
            acc = calc_accuracy(output, target)

            losses.update(loss.item(), B)
            accuracy.update(acc.item(), B)

    print('Loss {loss.avg:.4f}\t'
          'Acc: {acc.avg:.4f} \t'.format(loss=losses, acc=accuracy))
    args.writer_val.add_scalar('global/loss', losses.avg, epoch)
    args.writer_val.add_scalar('global/accuracy', accuracy.avg, epoch)

    return losses.avg, accuracy.avg


def test(dataset, model, criterion, device, epoch, args):
    # 10-crop then average the probability
    prob_dict = {}
    model.eval()

    # aug_list: 1,2,3,4,5 = top-left, top-right, bottom-left, bottom-right, center
    # flip_list: 0,1 = original, horizontal-flip
    if args.center_crop:
        print('Test using center crop')
        args.logger.log('Test using center_crop\n')
        aug_list = [5]; flip_list = [0]; title = 'center'
    if args.five_crop: 
        print('Test using 5 crop')
        args.logger.log('Test using 5_crop\n')
        aug_list = [5,1,2,3,4]; flip_list = [0]; title = 'five'
    if args.ten_crop:
        print('Test using 10 crop')
        args.logger.log('Test using 10_crop\n')
        aug_list = [5,1,2,3,4]; flip_list = [0,1]; title = 'ten'

    with torch.no_grad():
        end = time.time()
        for flip_idx in flip_list:
            for aug_idx in aug_list:
                print('Aug type: %d; flip: %d' % (aug_idx, flip_idx))
                if flip_idx == 0:
                    transform = transforms.Compose([
                                A.RandomHorizontalFlip(command='left'),
                                A.FiveCrop(size=(224,224), where=aug_idx),
                                A.Scale(size=(args.img_dim,args.img_dim)),
                                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3, consistent=True),
                                A.ToTensor(),
                                ])
                else:
                    transform = transforms.Compose([
                                A.RandomHorizontalFlip(command='right'),
                                A.FiveCrop(size=(224,224), where=aug_idx),
                                A.Scale(size=(args.img_dim,args.img_dim)),
                                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3, consistent=True),
                                A.ToTensor(),
                                ])

                dataset.transform = transform
                dataset.return_path = True
                dataset.return_label = True
                data_sampler = data.RandomSampler(dataset)
                data_loader = data.DataLoader(dataset,
                                              batch_size=1,
                                              sampler=data_sampler,
                                              shuffle=False,
                                              num_workers=16,
                                              pin_memory=True)


                for idx, (input_seq, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    B = 1
                    input_seq = input_seq.to(device)
                    target, vname = target
                    target = target.to(device)
                    input_seq = input_seq.squeeze(0) # squeeze the '1' batch dim
                    output, _ = model(input_seq)

                    prob_mean = nn.functional.softmax(output, 2).mean(1).mean(0, keepdim=True)

                    vname = vname[0]
                    if vname not in prob_dict.keys():
                        prob_dict[vname] = []
                    prob_dict[vname].append(prob_mean)

                # show intermediate result
                if (title == 'ten') and (flip_idx == 0) and (aug_idx == 5):
                    print('center-crop result:')
                    acc_1 = summarize_probability(prob_dict, 
                        data_loader.dataset.encode_action, 'center')
                    args.logger.log('center-crop:')
                    args.logger.log('test Epoch: [{0}]\t'
                        'Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
                        .format(epoch, acc=acc_1))

            # show intermediate result
            if (title == 'ten') and (flip_idx == 0):
                print('five-crop result:')
                acc_5 = summarize_probability(prob_dict, 
                        data_loader.dataset.encode_action, 'five')
                args.logger.log('five-crop:')
                args.logger.log('test Epoch: [{0}]\t'
                    'Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
                    .format(epoch, acc=acc_5))

        # show final result
        print('%s-crop result:' % title)
        acc_final = summarize_probability(prob_dict, 
            data_loader.dataset.encode_action, 'ten')
        args.logger.log('%s-crop:' % title)
        args.logger.log('test Epoch: [{0}]\t'
                        'Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
                        .format(epoch, acc=acc_final))
        sys.exit(0)


def summarize_probability(prob_dict, action_to_idx, title):
    acc = [AverageMeter(),AverageMeter()]
    stat = {}
    for vname, item in tqdm(prob_dict.items(), total=len(prob_dict)):
        try:
            action_name = vname.split('/')[-3]
        except:
            action_name = vname.split('/')[-2]
        target = action_to_idx(action_name)
        mean_prob = torch.stack(item, 0).mean(0)
        mean_top1, mean_top5 = calc_topk_accuracy(mean_prob, torch.LongTensor([target]).cuda(), (1,5))
        stat[vname] = {'mean_prob': mean_prob.tolist()}
        acc[0].update(mean_top1.item(), 1)
        acc[1].update(mean_top5.item(), 1)

    print('Mean: Acc@1: {acc[0].avg:.4f} Acc@5: {acc[1].avg:.4f}'
          .format(acc=acc))

    with open(os.path.join(os.path.dirname(args.test), 
        '%s-prob-%s.json' % (os.path.basename(args.test), title)), 'w') as fp:
        json.dump(stat, fp)
    return acc


def get_data(transform, mode='train'):
    print('Loading data for "%s" ...' % mode)
    global dataset
    if args.dataset == 'ucf101':
        dataset = UCF101Dataset(mode=mode, 
                         transform=transform, 
                         seq_len=args.seq_len,
                         num_seq=args.num_seq,
                         downsample=args.ds,
                         which_split=args.split,
                         return_label=True)
    elif args.dataset == 'hmdb51':
        dataset = HMDB51Dataset(mode=mode, 
                         transform=transform, 
                         seq_len=args.seq_len,
                         num_seq=args.num_seq,
                         downsample=args.ds,
                         which_split=args.split,
                         return_label=True)
    else:
        raise ValueError('dataset not supported')
    my_sampler = data.RandomSampler(dataset)
    if mode == 'train':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=args.workers,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'val':
        data_loader = data.DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=args.workers,
                                      pin_memory=True,
                                      drop_last=True)
    elif mode == 'test':
        data_loader = data.DataLoader(dataset,
                                      batch_size=1,
                                      sampler=my_sampler,
                                      shuffle=False,
                                      num_workers=args.workers,
                                      pin_memory=True)
    print('"%s" dataset size: %d' % (mode, len(dataset)))
    return data_loader, dataset


def set_path(args):
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    else:
        exp_path = 'log_{args.prefix}/{args.dataset}-{args.img_dim}-\
sp{args.split}_{args.net}_{args.model}_bs{args.batch_size}_\
lr{0}_wd{args.wd}_ds{args.ds}_seq{args.num_seq}_len{args.seq_len}_\
dp{args.dropout}_train-{args.train_what}{1}'.format(
                    args.old_lr if args.old_lr is not None else args.lr, \
                    '_pt='+args.pretrain.replace('/','-') if args.pretrain else '', \
                    args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): os.makedirs(img_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    return img_path, model_path


def MultiStepLR_Restart_Multiplier(epoch, gamma=0.1, step=[10,15,20], repeat=3):
    '''return the multipier for LambdaLR, 
    0  <= ep < 10: gamma^0
    10 <= ep < 15: gamma^1 
    15 <= ep < 20: gamma^2
    20 <= ep < 30: gamma^0 ... repeat 3 cycles and then keep gamma^2'''
    max_step = max(step)
    effective_epoch = epoch % max_step
    if epoch // max_step >= repeat:
        exp = len(step) - 1
    else:
        exp = len([i for i in step if effective_epoch>=i])
    return gamma ** exp


if __name__ == '__main__':
    args = parse_args()
    main(args)
