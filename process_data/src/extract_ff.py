import json
import os
import sys 
import cv2
import numpy as np 
from joblib import delayed, Parallel 
from tqdm import tqdm 
import platform
import argparse
import glob


def compute_TVL1(prev, curr, bound=20):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    flow = np.clip(flow, -bound, bound)

    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype('uint8')
    return flow


def extract_ff_opencv(v_path, frame_root, flow_root):
    '''opencv version: 
       v_path: single video path xxx/action/vname.mp4
       frame_root: root to store flow
       flow_root: root to store flow '''
    v_class = v_path.split('/')[-2]
    v_name = os.path.basename(v_path)[0:-4]
    frame_out_dir = os.path.join(frame_root, v_class, v_name)
    flow_out_dir = os.path.join(flow_root, v_class, v_name)
    for i in [frame_out_dir, flow_out_dir]:
        if not os.path.exists(i): 
            os.makedirs(i)
        else:
            print('[WARNING]', i, 'exists, continue...')
            

    vidcap = cv2.VideoCapture(v_path)
    nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    if len(glob.glob(os.path.join(frame_out_dir, '*.jpg'))) >= nb_frames - 3: # tolerance = 3 frame difference
        print('[WARNING]', frame_out_dir, 'has finished, dropped!')
        vidcap.release()
        return

    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    if (width == 0) or (height==0): 
        print(width, height, v_path)

    empty_img = 128 * np.ones((int(height),int(width),3)).astype(np.uint8)
    success, image = vidcap.read()
    count = 1

    pbar = tqdm(total=nb_frames)
    while success:
        cv2.imwrite(os.path.join(frame_out_dir, 'image_%05d.jpg' % count), 
                    image, 
                    [cv2.IMWRITE_JPEG_QUALITY, 100]) # quality from 0-100, 95 is default, high is good
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if count != 1:
            flow = compute_TVL1(prev_gray, image_gray)
            flow_img = empty_img.copy()
            flow_img[:,:,0:2] = flow
            cv2.imwrite(os.path.join(flow_out_dir, 'flow_%05d.jpg' % (count-1)),
                        flow_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])

        prev_gray = image_gray 
        success, image = vidcap.read()
        count += 1
        pbar.update(1)
    
    if nb_frames > count:
        print(frame_out_dir, 'is NOT extracted successfully', nb_frames, count)
    vidcap.release()

    return 



def main_UCF101(v_root, frame_root, flow_root):
    print('extracting UCF101 ... ')
    print('extracting videos from %s' % v_root)
    print('frame save to %s' % frame_root)
    print('flow save to %s' % flow_root)
    
    if not os.path.exists(frame_root): os.makedirs(frame_root)
    if not os.path.exists(flow_root): os.makedirs(flow_root)

    v_act_root = glob.glob(os.path.join(v_root, '*/'))
    for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
        v_paths = glob.glob(os.path.join(j, '*.avi'))
        v_paths = sorted(v_paths)
        Parallel(n_jobs=32)(delayed(extract_ff_opencv)\
            (p, frame_root, flow_root) for p in tqdm(v_paths, total=len(v_paths)))


def main_HMDB51(v_root, frame_root, flow_root):
    print('extracting HMDB51 ... ')
    print('extracting videos from %s' % v_root)
    print('frame save to %s' % frame_root)
    print('flow save to %s' % flow_root)
    
    if not os.path.exists(frame_root): os.makedirs(frame_root)
    if not os.path.exists(flow_root): os.makedirs(flow_root)

    v_act_root = glob.glob(os.path.join(v_root, '*/'))
    for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
        v_paths = glob.glob(os.path.join(j, '*.avi'))
        v_paths = sorted(v_paths)
        Parallel(n_jobs=32)(delayed(extract_ff_opencv)\
            (p, frame_root, flow_root) for p in tqdm(v_paths, total=len(v_paths)))


def main_kinetics400(v_root, frame_root, flow_root):
    print('extracting Kinetics400 ... ')
    for basename in ['train_split', 'val_split']:
        v_root_real = v_root + '/' + basename
        if not os.path.exists(v_root_real):
            print('Wrong v_root'); sys.exit()

        frame_root_real = os.path.join(frame_root, basename)
        flow_root_real = os.path.join(flow_root, basename)
        print('frame save to %s' % frame_root_real)
        print('flow save to %s' % flow_root_real)

        if not os.path.exists(frame_root_real): os.makedirs(frame_root_real)
        if not os.path.exists(flow_root_real): os.makedirs(flow_root_real)

        v_act_root = glob.glob(os.path.join(v_root_real, '*/'))
        v_act_root = sorted(v_act_root)

        # if resume, remember to delete the last video folder
        for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
            v_paths = glob.glob(os.path.join(j, '*.mp4'))
            v_paths = sorted(v_paths)

            # for resume:
            v_class = j.split('/')[-2]
            out_dir = os.path.join(frame_root_real, v_class)
            if os.path.exists(out_dir): 
                print(out_dir, 'exists!')
                continue

            print('extracting: %s' % v_class)
            Parallel(n_jobs=32)(delayed(extract_ff_opencv)\
                (p, frame_root_real, flow_root_real) for p in tqdm(v_paths, total=len(v_paths))) 


if __name__ == '__main__':
    # edit 'your_path' here: 
    main_UCF101(v_root='your_path/UCF101/videos',
                frame_root='your_path/UCF101/frame',
                flow_root='your_path/UCF101/flow')

    main_HMDB51(v_root='your_path/HMDB51/videos',
                frame_root='your_path/HMDB51/frame',
                flow_root='your_path/HMDB51/flow')

    main_kinetics400(v_root='your_path/Kinetics400/videos',
                frame_root='your_path/Kinetics400/frame',
                flow_root='your_path/Kinetics400/flow')