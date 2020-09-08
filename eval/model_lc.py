import math
import numpy as np
import sys
sys.path.append('../')
from backbone.select_backbone import select_resnet
from backbone.convrnn import ConvGRU

import torch
import torch.nn as nn
import torch.nn.functional as F

class LC(nn.Module):
    '''Linear Classifier'''
    def __init__(self, sample_size, num_seq, seq_len, 
                 network='resnet18', dropout=0.5, num_class=101, train_what='all'):
        super(LC, self).__init__()
        torch.cuda.manual_seed(666)
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.num_class = num_class 
        self.train_what = train_what

        print('=> Use 2D-3D %s backbone' % network)
        self.last_duration = int(math.ceil(seq_len / 4))
        self.last_size = int(math.ceil(sample_size / 32))

        self.backbone, self.param = select_resnet(network)
        self.param['num_layers'] = 1
        self.param['hidden_size'] = self.param['feature_size']

        self.agg_f = ConvGRU(input_size=self.param['feature_size'],
                           hidden_size=self.param['hidden_size'],
                           kernel_size=1,
                           num_layers=self.param['num_layers'])
        self.agg_b = ConvGRU(input_size=self.param['feature_size'],
                           hidden_size=self.param['hidden_size'],
                           kernel_size=1,
                           num_layers=self.param['num_layers'])
        self._initialize_weights(self.agg_f)
        self._initialize_weights(self.agg_b)

        self.final_bn = nn.BatchNorm1d(self.param['feature_size']*2)
        self.final_bn.weight.data.fill_(1)
        self.final_bn.bias.data.zero_()

        self.final_fc = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(self.param['feature_size']*2, self.num_class))
        self._initialize_weights(self.final_fc)

    def forward(self, block):
        # seq1: [B, N, C, SL, W, H]
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B*N, C, SL, H, W)
        enable_grad = self.train_what!='last'
        with torch.set_grad_enabled(enable_grad):
            feature = self.backbone(block)
            feature = F.relu(feature)
            feature = F.avg_pool3d(feature, (self.last_duration, 1, 1), stride=1)
            feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # [B*N,D,last_size,last_size]

            context_forward, _ = self.agg_f(feature)
            context_forward = context_forward[:,-1,:].unsqueeze(1)
            context_forward = F.avg_pool3d(context_forward, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1)

            feature_back = torch.flip(feature, dims=(1,))
            context_back, _ = self.agg_b(feature_back)
            context_back = context_back[:,-1,:].unsqueeze(1)
            context_back = F.avg_pool3d(context_back, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1)

            context = torch.cat([context_forward, context_back], dim=-1) # B,N,C=2C

        context = self.final_bn(context.transpose(-1,-2)).transpose(-1,-2) # [B,N,C] -> [B,C,N] -> BN() -> [B,N,C], because BN operates on id=1 channel.
        output = self.final_fc(context).view(B, -1, self.num_class)

        return output, context

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)        


