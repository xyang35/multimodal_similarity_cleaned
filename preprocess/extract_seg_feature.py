# Extract spatial pyramid features of semantic segmentation
# Reference: Semantic segmentation as image representation for scene recognition

import glob
import os
import pdb
import numpy as np
import copy


with open('/mnt/work/honda_100h/test_session.txt', 'r') as fin:
    session_ids = fin.read().strip().split('\n')

def softmax(x):    # important to do in-place for large array
    #return np.exp(x-np.max(x,axis=-1,keepdims=True)) / \
    #        np.sum(np.exp(x-np.max(x,axis=-1,keepdims=True)),axis=-1,keepdims=True)
    temp = np.max(x,axis=-1,keepdims=True)
    res = np.subtract(x, temp, out=x)    # do subtraction in-place
    del temp
    res = np.exp(x, x)    # do exp in-place
    res = np.divide(x, np.sum(x, axis=-1, keepdims=True), x)    # do divide in-place

    return x

seg_root = '/mnt/data/honda_100h_archive/semantic_segmentation/'
feat_root = '/mnt/work/honda_100h/features/'

L = 3    # number of pyramid levels

files = glob.glob(seg_root+'*.npy')
for i, f in enumerate(files):
    session_id = os.path.basename(f).replace('_seg.npy',"")
    if not session_id in session_ids:
        continue
    
    output_name = os.path.basename(f).replace('.npy', '_sp.npy')
    print (i, '/', len(files)-1, ":", output_name)
    if os.path.isfile(feat_root+output_name):
        continue

    seg = np.load(f)
    N, H, W, D = seg.shape
    print (N)
    seg = softmax(seg)

    print ('here')
    feat = []
    # for each level
    for l in range(L):
        h_size = H // (2**l)
        w_size = W // (2**l)
        # for each bin
        for i in range(2**l):
            for j in range(2**l):
                region = copy.copy(seg[:, i*h_size:(i+1)*h_size, j*w_size:(j+1)*w_size, :])
                # get histogram (soft) within a bin
                feat.append(np.mean(region, axis=(1,2)))
                del region

    del seg
    print ('here')

    # concatenate features of different levels and bins
    feat = np.concatenate(feat, axis=1)
    np.save(feat_root+output_name, feat)

