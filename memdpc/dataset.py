import torch
from torch.utils import data
from torchvision import transforms
import os
import sys
import time
import pickle
import glob
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

def read_file(path):
    with open(path, 'r') as f:
        content = f.readlines()
    content = [i.strip() for i in content]
    return content

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class K400Dataset(data.Dataset):
    def __init__(self,
                 root='%s/../process_data/data/k400' % os.path.dirname(os.path.abspath(__file__)),
                 mode='val',
                 transform=None,
                 seq_len=5,
                 num_seq=8,
                 downsample=3,
                 return_label=False):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.return_label = return_label

        classes = read_file(os.path.join(root, 'ClassInd.txt'))
        print('Frame Dataset from {} has #class {}'.format(root, len(classes)))
        self.num_class = len(classes)
        self.class_to_idx = {classes[i]:i for i in range(len(classes))}
        self.idx_to_class = {i:classes[i] for i in range(len(classes))}

        # splits
        if mode == 'train':
            split = '../process_data/data/kinetics400/train_split.csv'
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            split = '../process_data/data/kinetics400/val_split.csv'
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        drop_idx = []
        print('filter out too short videos ...')
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info)):
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                drop_idx.append(idx) 
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': 
            self.video_info = self.video_info.sample(frac=0.3, random_state=666)

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        start_idx = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), 1)
        seq_idx = np.arange(self.num_seq*self.seq_len)*self.downsample + start_idx
        return seq_idx

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        frame_index = self.idx_sampler(vlen, vpath)
        
        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in frame_index]
        t_seq = self.transform(seq) 
        
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)

            label = torch.LongTensor([vid])
            return t_seq, label

        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        return self.class_to_idx[action_name]

    def decode_action(self, action_code):
        return self.idx_to_class[action_code]


class UCF101Dataset(data.Dataset):
    def __init__(self,
                 root='%s/../process_data/data/ucf101' % os.path.dirname(os.path.abspath(__file__)),
                 mode='val',
                 transform=None, 
                 seq_len=5,
                 num_seq=8,
                 downsample=3,
                 which_split=1,
                 return_label=False):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.which_split = which_split
        self.return_label = return_label

        # splits
        if mode == 'train':
            split = '../process_data/data/ucf101/train_split%02d.csv' % self.which_split
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'): # use val for test
            split = '../process_data/data/ucf101/test_split%02d.csv' % self.which_split 
            video_info = pd.read_csv(split, header=None)
        else: raise ValueError('wrong mode')

        # get action list
        classes = read_file(os.path.join(root, 'ClassInd.txt'))
        print('Frame Dataset from {} has #class {}'.format(root, len(classes)))
        self.num_class = len(classes)
        self.class_to_idx = {classes[i]:i for i in range(len(classes))}
        self.idx_to_class = {i:classes[i] for i in range(len(classes))}

        # filter out too short videos:
        drop_idx = []
        for idx, row in video_info.iterrows():
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                drop_idx.append(idx)
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': 
            self.video_info = self.video_info.sample(frac=0.3)

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        start_idx = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), 1)
        seq_idx = np.arange(self.num_seq*self.seq_len)*self.downsample + start_idx
        return seq_idx


    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        frame_index = self.idx_sampler(vlen, vpath)
        
        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in frame_index]
        t_seq = self.transform(seq)
        
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)
            label = torch.LongTensor([vid])
            return t_seq, label
            
        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        return self.class_to_idx[action_name]

    def decode_action(self, action_code):
        return self.idx_to_class[action_code]

