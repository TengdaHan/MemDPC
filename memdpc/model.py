import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../')
from backbone.select_backbone import select_resnet
from backbone.convrnn import ConvGRU
from utils.utils import calc_topk_accuracy


class MemDPC_BD(nn.Module):
    '''MemDPC with bi-directional RNN'''
    def __init__(self, 
                 sample_size, 
                 num_seq=8, 
                 seq_len=5, 
                 pred_step=3, 
                 network='resnet18', 
                 mem_size=1024):
        super(MemDPC_BD, self).__init__()
        print('Using MemDPC-BiDirectional model with {} and mem_size {}'\
            .format(network, mem_size))
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = pred_step
        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 32))
        self.mem_size = mem_size
        self.tgt_dict = {}
        print('final feature map has size %dx%d' % (self.last_size, self.last_size))

        self.backbone, self.param = select_resnet(network)
        self.param['num_layers'] = 1 # param for GRU
        self.param['hidden_size'] = self.param['feature_size'] # param for GRU
        self.param['membanks_size'] = mem_size
        self.mb = torch.nn.Parameter(torch.randn(self.param['membanks_size'], self.param['feature_size']))
        print('MEM Bank has size %dx%d' % (self.param['membanks_size'], self.param['feature_size']))

        # bi-directional RNN
        self.agg_f = ConvGRU(input_size=self.param['feature_size'],
                             hidden_size=self.param['hidden_size'],
                             kernel_size=1,
                             num_layers=self.param['num_layers'])
        self.agg_b = ConvGRU(input_size=self.param['feature_size'],
                             hidden_size=self.param['hidden_size'],
                             kernel_size=1,
                             num_layers=self.param['num_layers'])

        self.network_pred = nn.Sequential(
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.param['feature_size'], self.param['membanks_size'], kernel_size=1, padding=0)
                                )
        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        self._initialize_weights(self.agg_f)
        self._initialize_weights(self.agg_b)
        self._initialize_weights(self.network_pred)


    def get_loss(self, pred, gt, B, SL, last_size, feature_size, kernel=1):
        # pred: B,C,N,H,W
        # GT: C,B,N,H*H
        score = torch.matmul(pred.permute(0,2,3,4,1).contiguous().view(B*SL*last_size**2,feature_size),
                             gt.contiguous().view(feature_size, B*SL*last_size**2)) 
        if SL not in self.tgt_dict:
            self.tgt_dict[SL] = torch.arange(B*SL*last_size**2)
        tgt = self.tgt_dict[SL].to(score.device)
        loss = self.ce_loss(score, tgt)
        top1, top5 = calc_topk_accuracy(score, tgt, (1,5))
        return loss, top1, top5

        
    def forward(self, block):
        # extract feature
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B*N, C, SL, H, W)
        feat3d = self.backbone(block)

        feat3d = F.avg_pool3d(feat3d, (self.last_duration, 1, 1), stride=(1, 1, 1))
        feat3d = feat3d.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # before ReLU, (-inf, +inf)

        losses = [] # all loss
        acc = [] # all acc
        loss = 0
        gt = feat3d.permute(2,0,1,3,4).contiguous().view(self.param['feature_size'], B, N, self.last_size**2)

        feat3d_b = torch.flip(feat3d, dims=(1,))
        gt_b = torch.flip(gt, dims=(2,))

        # forward MemDPC
        pd_tmp_pool = []
        for j in range(self.pred_step):
            if j == 0:
                feat_tmp = feat3d[:,0:(N-self.pred_step),:,:,:]
                _, hidden = self.agg_f(F.relu(feat_tmp))
                context_feature = hidden.clone()
            else:
                _, hidden = self.agg_f(F.relu(pd_tmp).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:,-1,:] # after tanh, (-1,1). get the hidden state of last layer, last time step
            pd_tmp = self.network_pred(hidden)
            pd_tmp = F.softmax(pd_tmp, dim=1) # B,MEM,H,W
            pd_tmp = torch.einsum('bmhw,mc->bchw', pd_tmp, self.mb)
            pd_tmp_pool.append(pd_tmp)

        pd_tmp_pool = torch.stack(pd_tmp_pool, dim=2); SL_tmp = pd_tmp_pool.size(2)
        gt_tmp = gt[:,:,-self.pred_step::,:]
        loss_tmp, top1, top5 = self.get_loss(pd_tmp_pool, gt_tmp, B, SL_tmp, self.last_size, self.param['feature_size'])
        loss_tmp = loss_tmp.mean()
        loss = loss_tmp
        losses.append(loss_tmp.data.unsqueeze(0))
        acc.append(torch.stack([top1, top5], 0).unsqueeze(0))


        # backward MemDPC
        pd_tmp_pool_b = []
        for j in range(self.pred_step):
            if j == 0:
                feat_tmp = feat3d_b[:,0:(N-self.pred_step),:,:,:]
                _, hidden = self.agg_b(F.relu(feat_tmp))
            else:
                _, hidden = self.agg_b(F.relu(pd_tmp_b).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:,-1,:] # after tanh, (-1,1). get the hidden state of last layer, last time step
            pd_tmp_b = self.network_pred(hidden)
            pd_tmp_b = F.softmax(pd_tmp_b, dim=1) # B,MEM,H,W
            pd_tmp_b = torch.einsum('bmhw,mc->bchw', pd_tmp_b, self.mb)
            pd_tmp_pool_b.append(pd_tmp_b)

        pd_tmp_pool_b = torch.stack(pd_tmp_pool_b, dim=2); SL_tmp = pd_tmp_pool_b.size(2)
        gt_tmp_b = gt_b[:,:,-self.pred_step::,:]
        loss_tmp_b, top1, top5 = self.get_loss(pd_tmp_pool_b, gt_tmp_b, B, SL_tmp, self.last_size, self.param['feature_size'])
        loss_tmp_b = loss_tmp_b.mean()
        losses.append(loss_tmp_b.data.unsqueeze(0))
        acc.append(torch.stack([top1, top5], 0).unsqueeze(0))

        loss = loss + loss_tmp_b

        return loss, losses, acc, context_feature


    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 0.1)

