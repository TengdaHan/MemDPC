# by htd@robots.ox.ac.uk
from joblib import delayed, Parallel
import os 
import sys 
import glob 
import subprocess
from tqdm import tqdm 
import cv2
import numpy as np 
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def resize_video_ffmpeg(v_path, out_path, dim=256):
    '''v_path: single video path;
       out_path: root to store output videos'''
    v_class = v_path.split('/')[-2]
    v_name = os.path.basename(v_path)[0:-4]
    out_dir = os.path.join(out_path, v_class)
    if not os.path.exists(out_dir):
        raise ValueError("directory not exist, it shouldn't happen")

    vidcap = cv2.VideoCapture(v_path)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    if (width == 0) or (height==0): 
        print(v_path, 'not successfully loaded, drop ..'); return
    new_dim = resize_dim(width, height, dim)
    if new_dim[0] == dim: 
        dim_cmd = '%d:-2' % dim
    elif new_dim[1] == dim:
        dim_cmd = '-2:%d' % dim 

    cmd = ['ffmpeg', '-loglevel', 'quiet', '-y',
           '-i', '%s'%v_path,
           '-vf',
           'scale=%s'%dim_cmd,
           '%s' % os.path.join(out_dir, os.path.basename(v_path))]
    ffmpeg = subprocess.call(cmd)

def resize_dim(w, h, target):
    '''resize (w, h), such that the smaller side is target, keep the aspect ratio'''
    if w >= h:
        return [int(target * w / h), int(target)]
    else:
        return [int(target), int(target * h / w)]

def main_kinetics400(output_path='your_path/kinetics400'):
    print('save to %s ... ' % output_path)
    for splitname in ['val_split', 'train_split']:
        v_root = '/datasets/KineticsVideo' + '/' + splitname
        if not os.path.exists(v_root):
            print('Wrong v_root')
            import ipdb; ipdb.set_trace() # for debug
        out_path = os.path.join(output_path, splitname) 
        if not os.path.exists(out_path): 
            os.makedirs(out_path)
        v_act_root = glob.glob(os.path.join(v_root, '*/'))
        v_act_root = sorted(v_act_root)

        # if resume, remember to delete the last video folder
        for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
            v_paths = glob.glob(os.path.join(j, '*.mp4'))
            v_paths = sorted(v_paths)
            v_class = j.split('/')[-2]
            out_dir = os.path.join(out_path, v_class)
            if os.path.exists(out_dir): 
                print(out_dir, 'exists!'); continue
            else:
                os.makedirs(out_dir)

            print('extracting: %s' % v_class)
            Parallel(n_jobs=8)(delayed(resize_video_ffmpeg)(p, out_path, dim=256) for p in tqdm(v_paths, total=len(v_paths)))



if __name__ == '__main__':
    main_kinetics400(output_path='your_path/kinetics400')
    # users need to change output_path and v_root
