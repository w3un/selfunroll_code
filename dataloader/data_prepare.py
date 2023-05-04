from pathlib import Path
from re import L
import h5py
# from util.eventslicer import EventSlicer
import numpy as np
import os
from tqdm import tqdm
import shutil
import cv2
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def filter_events_by_space(key, x1, x2, x3, start, end):
    ## filter events based on spatial dimension
    # start inclusive and end exclusive
    new_x1 = x1[key >= start]
    new_x2 = x2[key >= start]
    new_x3 = x3[key >= start]
    new_key = key[key >= start]

    new_x1 = new_x1[new_key < end]
    new_x2 = new_x2[new_key < end]
    new_x3 = new_x3[new_key < end]
    new_key = new_key[new_key < end]

    return new_key, new_x1, new_x2, new_x3


parent_dir = '/home/wyg/Documents/wyg/my_simulate/test'
dataset_path = Path(parent_dir)
in_t_us = 1 / 13171.
dir_list = []
for child in dataset_path.iterdir():
    child = str(child)
    dir_list.append(child)
# parent_dir = '/home/wyg/Documents/wyg/my_simulate/train'
# dataset_path = Path(parent_dir)
# in_t_us = 1 / 13171.
# for child in dataset_path.iterdir():
#     child = str(child)
#     dir_list.append(child)
# dir_list =dir_list[:6]
dir_list.sort()
counter = 1
print(dir_list)
dir_list.reverse()
for child in dir_list:

    # child = '/home/wyg/Documents/my_simulate/test/24209_1_38 125'
    print(child)

    gs_dir = Path(child) / 'gs'
    gs_file = list()

    for entry in gs_dir.iterdir():
        assert str(entry.name).endswith('.png')
        gs_file.append(str(entry))
    gs_file.sort()
    gslen = len(gs_file)
    mkdir(str(Path(child)/'newevunroll'))
    mkdir(str(Path(child) / 'newrs'))
    h5file = child + '/'+child.split('/')[-1] + '.h5'
    with h5py.File(h5file, 'r') as f:
        events = f['events'][:,]
        t = events[:,0]
        x = events[:, 1]
        y = events[:, 2]
        p = events[:, 3]
        x = x.astype(np.int64)
        y = y.astype(np.int64)
        p = p.astype(np.int64)
    if str(child).split('\\')[-1] in ['24209_1_33', '24209_1_30', '24209_1_31', '24209_1_4', '24209_1_24',
                                     '24209_1_41']:
        # exp = 73
        delay = 180
        img_len = gslen // delay
    else:
        # exp = 145
        delay = 360
        img_len = gslen // delay
    tar = tqdm(total=img_len, position=0, leave=True)
    for index in range(img_len):
        total_exp = delay
        starttime = index*delay
        T,X,Y,P = filter_events_by_space(t,x,y,p,starttime/5000*1e6,(starttime+total_exp)/5000*1e6)
        total_event = {'x': X, 'y': Y, 'p': P, 't': T}
        np.savez(os.path.join(str(child), 'newevunroll', '%03d.npz' % index), x=X, y=Y, p=P, t=T)
        # generate rs image
        temp = np.zeros((360,640,3)).astype(np.uint8)
        for i in range(delay):
            gsi = cv2.imread(gs_file[i + index * delay])
            if delay==360:
                temp[i]= gsi[i]
            if delay==180:
                temp[2*i:2*i+2] = gsi[2*i:2*i+2]
        cv2.imwrite(os.path.join(str(child), 'newrs', '%03d.png' % index), temp.astype(np.uint8))

        tar.update(1)