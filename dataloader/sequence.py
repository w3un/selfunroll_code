from pathlib import Path

import numpy as np
from torch.utils.data import IterableDataset, Dataset
import torchvision.transforms.functional as F
from PIL import Image
from my_util import *
import torch
import cv2
import h5py

class sequence_gev(Dataset):
    def __init__(self, seq_path: Path, mode: str = 'train', target=0.5, num_bins=16, crop_sz_W=256,
                 crop_sz_H=256, noise=0.1):
        assert num_bins >= 1
        assert seq_path.is_dir()
        self.seq_path = seq_path
        # NOTE: Adapt this code according to the present mode (e.g. train, val or test).
        self.mode = mode
        self.height = 360
        self.width = 640
        self.target = target
        if self.mode == 'train':

            self.crop_sz_W = crop_sz_W
            self.crop_sz_H = crop_sz_H
        else:
            self.crop_sz_W = self.width
            self.crop_sz_H = self.height
        self.reverse = 0
        # Save output dimensions

        self.num_bins = num_bins
        self.num_frame = num_bins
        # self.num_frame
        self.noise = noise
        # self.target = target
        # Save interval timestamp in ms
        self.in_t_us = 1e6 / 5000
        # load RS  image
        RS_dir = seq_path / 'rs'
        assert RS_dir.is_dir()
        RS_file = list()
        for entry in RS_dir.iterdir():
            assert str(entry.name).endswith('.png')
            RS_file.append(str(entry))
        RS_file.sort()
        self.RS_file = RS_file
        # load event
        for entry in seq_path.iterdir():
            if str(entry.name).endswith('.h5'):
                evnet_file = str(entry)
        self.evnet_file = evnet_file
        subev_dir = seq_path / 'evunroll'
        assert subev_dir.is_dir()
        subev_file = list()
        for entry in subev_dir.iterdir():
            if str(entry.name).endswith('npz'):
                subev_file.append(str(entry))
        subev_file.sort()
        self.subev_file = subev_file
        #
        self.gap = 100
        if str(self.seq_path).split('/')[-1] in ['24209_1_33', '24209_1_30', '24209_1_31', '24209_1_4', '24209_1_24',
                                                 '24209_1_41']:
            self.rslen  = 180
            self.exp = 8
        else:
            self.rslen = 360
            self.exp = 16
        # load GT
        GT_dir = seq_path / 'gs'
        assert GT_dir.is_dir()
        GT_file = list()
        for entry in GT_dir.iterdir():
            assert str(entry.name).endswith('.png')
            GT_file.append(str(entry))
        GT_file.sort()
        self.GT_file = GT_file

    @staticmethod
    def get_img_file(filepath: Path):
        assert filepath.is_file()
        img = Image.open(str(filepath))
        img = np.asarray(img)
        img = F.to_tensor(img)
        return img

    def __len__(self):
        return len(self.RS_file) - 1

    def __getitem__(self, index):

        # start_time = int(time.time())

        RS_path = Path(self.RS_file[index])
        rs1 = self.get_img_file(RS_path)
        RS_path = Path(self.RS_file[index + 1])
        rs2 = self.get_img_file(RS_path)
        C, H, W = rs1.shape

        if self.mode == 'train':
            cx = np.random.randint(0, self.width - self.crop_sz_W)
            cy = np.random.randint(0, self.height - self.crop_sz_H)
        else:
            cx = int((self.width - self.crop_sz_W) / 2)
            cy = int((self.height - self.crop_sz_H) / 2)
        if str(self.seq_path).split('/')[-1] in ['24209_1_33', '24209_1_30', '24209_1_31', '24209_1_4', '24209_1_24',
                                                 '24209_1_41']:
            exp = 8
            rs1_start = [self.in_t_us * self.gap * index + self.in_t_us * i / 2 for i in range(2 * H)]
            rs1_end = [self.in_t_us * self.gap * index + self.in_t_us * i / 2 + self.in_t_us * exp for i in range(H)]
            rs2_start = [self.in_t_us * self.gap * index + self.in_t_us * self.gap + self.in_t_us * i / 2 for i in range(H)]
            rs2_end = [self.in_t_us * self.gap * index + self.in_t_us * self.gap + self.in_t_us * i / 2 + self.in_t_us * exp for i
                       in range(H)]
            # select1 = np.random.randint(180, 280)
            if self.mode == 'train':
                select1 = np.random.randint(252 - self.gap, 252 + self.gap)
            else:
                select1 = int(360 * self.target)
            target_exp = self.in_t_us * 8
            gs_path = Path(self.GT_file[self.gap * index + int(select1 / 2)])  #
            gs = self.get_img_file(gs_path)

            target = [rs1_start[select1] for i in range(H)]

        else:
            exp = 16
            delay = 360
            rs1_start = [self.in_t_us * self.gap * index + self.in_t_us * i for i in range(H + 180)]
            rs1_end = [self.in_t_us * self.gap * index + self.in_t_us * i + self.in_t_us * exp for i in range(H + 180)]
            rs2_start = [self.in_t_us * self.gap * index + self.in_t_us * self.gap + self.in_t_us * i for i in range(H)]
            rs2_end = [self.in_t_us * self.gap * index + self.in_t_us * self.gap + self.in_t_us * i + self.in_t_us * exp for i in
                       range(H)]
            # select1 = np.random.randint(180, 280)
            if self.mode == 'train':
                select1 = np.random.randint(252 - self.gap, 252 + self.gap)
            else:
                select1 = int(360 * self.target)
            target_exp = self.in_t_us * 16
            gs_path = Path(self.GT_file[self.gap * index + select1])  #
            gs = self.get_img_file(gs_path)
            target = [rs1_start[select1] for i in range(H)]

        target = np.asarray(target)
        rs1_start = np.asarray(rs1_start)
        rs1_end = np.asarray(rs1_end)
        rs2_start = np.asarray(rs2_start)
        rs2_end = np.asarray(rs2_end)
        # if self.mode == 'train':
        target = target[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy # - self.in_t_us * cy
        rs1_start = rs1_start[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy
        rs1_end = rs1_end[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy
        rs2_start = rs2_start[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy
        rs2_end = rs2_end[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy
        # print(self.subev_file[index])


        with h5py.File(self.evnet_file, 'r') as f:
            events = f['events']
            img_to_idx = f['img_to_idx']
            start_idx,end_idx = img_to_idx[index*self.gap],img_to_idx[(index+1)*self.gap+self.rslen+self.exp]
            event_input = events[start_idx:end_idx,:].astype(np.int64)
        # event1 = np.load(self.subev_file[index])
        # x1, y1, p1, t1 = event1['x'], event1['y'], event1['p'], event1['t']
        # x1 = x1.astype(np.int64)
        # y1 = y1.astype(np.int64)
        # p1 = p1.astype(np.int64)
        # event2 = np.load(self.subev_file[index + 1])
        # x2, y2, p2, t2 = event2['x'], event2['y'], event2['p'], event2['t']
        # x2 = x2.astype(np.int64)
        # y2 = y2.astype(np.int64)
        # p2 = p2.astype(np.int64)
        # # t1_, y1_, p1_, x1_ = filter_events_by_space(t1, y1, p1, x1,
        # #                                             self.in_t_us * 100 * index + self.in_t_us * 100,
        # #                                             self.in_t_us * 100 * index + self.in_t_us * delay + self.in_t_us * exp)
        #
        # t2, y2, p2, x2 = filter_events_by_space(t2, y2, p2, x2,
        #                                         rs1_end[-1],
        #                                         100000000)
        # x = np.concatenate((x1, x2), 0)
        # y = np.concatenate((y1, y2), 0)
        # p = np.concatenate((p1, p2), 0)
        # t = np.concatenate((t1, t2), 0)
        # if self.mode == 'train':
        x,y,p,t = event_input[:,1],event_input[:,2],event_input[:,3],event_input[:,0]
        x, y, p, t = filter_events_by_space(x, y, p, t, cx, cx + self.crop_sz_W)
        y, x, p, t = filter_events_by_space(y, x, p, t, cy, cy + self.crop_sz_H)
        x = x - cx
        y = y - cy

        total_event = {'x': x, 'y': y, 'p': p, 't': t}
        # load_ev_time = int(time.time())
        gs_to_rs1_start_frame, gs_to_rs1_end_frame = event2frame(total_event,
                                                                 imgsize=[
                                                                     self.crop_sz_H,
                                                                     self.crop_sz_W],
                                                                 low_limit=target,
                                                                 high_limit1=rs1_start,
                                                                 high_limit2=rs1_end,
                                                                 num_frame=self.num_frame,
                                                                 noise=self.noise)
        target_exp_frame = np.zeros((self.num_frame, 2, self.crop_sz_H, self.crop_sz_W))
        target_exp_frame = e2f_detail(total_event, target_exp_frame, target, target + target_exp, self.noise)
        target_exp_frame = target_exp_frame.reshape(2 * self.num_frame, self.crop_sz_H, self.crop_sz_W)
        rs1_to_gs_start_frame, rs1_to_gs_end_frame = event2frame(total_event,
                                                                 imgsize=[
                                                                     self.crop_sz_H,
                                                                     self.crop_sz_W],
                                                                 low_limit=rs1_start,
                                                                 high_limit1=target,
                                                                 high_limit2=target + target_exp,
                                                                 num_frame=self.num_frame,
                                                                 noise=self.noise)
        rs1_exp_frame = np.zeros((self.num_frame, 2, self.crop_sz_H, self.crop_sz_W))
        rs1_exp_frame = e2f_detail(total_event, rs1_exp_frame, rs1_start, rs1_end, self.noise)
        rs1_exp_frame = rs1_exp_frame.reshape(2 * self.num_frame, self.crop_sz_H, self.crop_sz_W)
        gs_to_rs2_start_frame, gs_to_rs2_end_frame = event2frame(total_event,
                                                                 imgsize=[
                                                                     self.crop_sz_H,
                                                                     self.crop_sz_W],
                                                                 low_limit=target,
                                                                 high_limit1=rs2_start,
                                                                 high_limit2=rs2_end,
                                                                 num_frame=self.num_frame,
                                                                 noise=self.noise)

        rs2_to_gs_start_frame, rs2_to_gs_end_frame = event2frame(total_event,
                                                                 imgsize=[
                                                                     self.crop_sz_H,
                                                                     self.crop_sz_W],
                                                                 low_limit=rs2_start,
                                                                 high_limit1=target,
                                                                 high_limit2=target + target_exp,
                                                                 num_frame=self.num_frame,
                                                                 noise=self.noise)
        rs1_to_rs2_start_frame, rs1_to_rs2_end_frame = event2frame(total_event,
                                                                   imgsize=[
                                                                       self.crop_sz_H,
                                                                       self.crop_sz_W],
                                                                   low_limit=rs1_start,
                                                                   high_limit1=rs2_start,
                                                                   high_limit2=rs2_end,
                                                                   num_frame=self.num_frame,
                                                                   noise=self.noise)
        rs2_to_rs1_start_frame, rs2_to_rs1_end_frame = event2frame(total_event,
                                                                   imgsize=[
                                                                       self.crop_sz_H,
                                                                       self.crop_sz_W],
                                                                   low_limit=rs2_start,
                                                                   high_limit1=rs1_start,
                                                                   high_limit2=rs1_end,
                                                                   num_frame=self.num_frame,
                                                                   noise=self.noise)
        rs2_exp_frame = np.zeros((self.num_frame, 2, self.crop_sz_H, self.crop_sz_W))
        rs2_exp_frame = e2f_detail(total_event, rs2_exp_frame, rs2_start, rs2_end, self.noise)
        rs2_exp_frame = rs2_exp_frame.reshape(2 * self.num_frame, self.crop_sz_H, self.crop_sz_W)

        rs1 = rs1[..., cy:cy + self.crop_sz_H, cx:cx + self.crop_sz_W]
        rs2 = rs2[..., cy:cy + self.crop_sz_H, cx:cx + self.crop_sz_W]
        gt = gs[..., cy:cy + self.crop_sz_H, cx:cx + self.crop_sz_W]
        select = np.random.randint(0, len(self.GT_file))
        gs_path = Path(self.GT_file[select])
        gs = self.get_img_file(gs_path)
        gs = gs[..., cy:cy + self.crop_sz_H, cx:cx + self.crop_sz_W]

        out = {
            'file_index': self.RS_file[index],
            'Orig1': rs1,
            'Orig2': rs2,
            'TtoO1_sf': gs_to_rs1_start_frame,
            'TtoO1_ef': gs_to_rs1_end_frame,
            'TtoO2_sf': gs_to_rs2_start_frame,
            'TtoO2_ef': gs_to_rs2_end_frame,
            'Texpf': target_exp_frame,
            'O1toT_sf': rs1_to_gs_start_frame,
            'O1toT_ef': rs1_to_gs_end_frame,
            'O1expf': rs1_exp_frame,
            'O2toT_sf': rs2_to_gs_start_frame,
            'O2toT_ef': rs2_to_gs_end_frame,
            'O2expf': rs2_exp_frame,
            'O1toO2_sf': rs1_to_rs2_start_frame,
            'O1toO2_ef': rs1_to_rs2_end_frame,
            'O2toO1_sf': rs2_to_rs1_start_frame,
            'O2toO1_ef': rs2_to_rs1_end_frame,
            'GT': gt,
            'gs': gs
        }
        return out


class sequence_gevreal(Dataset):
    def __init__(self, seq_path: Path, target=0.5, mode: str = 'train', num_bins=16, crop_sz_W=256,
                 crop_sz_H=256, noise=0.1):
        assert num_bins >= 1
        assert seq_path.is_dir()
        self.seq_path = seq_path
        self.mode = mode
        self.target = target
        self.reverse = 0
        # Save output dimensions
        self.height = 360
        self.width = 640
        self.num_bins = num_bins
        self.num_frame = num_bins
        self.noise = noise
        # self.target = target
        # Save interval timestamp in ms
        self.in_t_us = 1 / 13171.
        # load RS  image
        self.fps = [20.79, 20.79, 20.79, 20.79, 20.79, 20.79, 20.79, 20.79, 20.79, 19.61, 19.61, 20.79, 20.79, 20.79,
                    20.79,
                    20.79]
        RS_dir = seq_path / 'image'
        assert RS_dir.is_dir()
        RS_file = list()
        for entry in RS_dir.iterdir():
            assert str(entry.name).endswith('.png')
            RS_file.append(str(entry))
        RS_file.sort()
        self.RS_file = RS_file
        infolist = self.seq_path.name.split('_')
        self.folder = int(infolist[0])
        if self.folder < 7:
            self.height = 260
            self.width = 346
        else:
            self.height = 720
            self.width = 1280
        if self.mode == 'train':
            self.crop_sz_W = crop_sz_W
            self.crop_sz_H = crop_sz_H
        else:
            self.crop_sz_W = self.width
            self.crop_sz_H = self.height
        for entry in self.seq_path.iterdir():
            if str(entry.name).endswith('npz'):
                self.ev_file = str(entry)
        self.event = np.load(self.ev_file)
        if self.folder >= 7:
            print(self.seq_path)
            self.x, self.y, self.p, self.t = self.event['event']['x'], self.event['event']['y'], self.event['event'][
                'p'], self.event['event']['t']
        else:
            print(self.seq_path)
            self.x = self.event['event'][:, 1]
            self.y = self.event['event'][:, 2]
            self.p = self.event['event'][:, 3]
            self.t = self.event['event'][:, 0]
        # if self.mode == 'train':
        self.x = self.x.astype(np.int64)
        self.y = self.y.astype(np.int64)
        self.p = self.p.astype(np.int64)
        self.t = self.t.astype(np.int64)
        self.p[self.p == -1] = 0
        self.exp_time = float(infolist[1])
        self.delay_time = float(infolist[2]) / float(self.height - 1)

    @staticmethod
    def get_img_file(filepath: Path):

        assert filepath.is_file()
        img_input = cv2.cvtColor(cv2.imread(str(filepath)), cv2.COLOR_BGR2RGB)
        img_input = torch.from_numpy(img_input.copy()).permute(2, 0, 1).float() / 255
        return img_input

    def __len__(self):

        return int((len(self.RS_file) - 1) )

    def __getitem__(self, index):
        # index +=48
        if self.mode == 'train':
            index=index+self.test_len
        RS_path = Path(self.RS_file[index])
        rs1 = self.get_img_file(RS_path)
        RS_path = Path(self.RS_file[index + 1])
        rs2 = self.get_img_file(RS_path)
        C, H, W = rs1.shape
        # self.height = H
        # self.width = W
        if self.mode == 'train':
            cx = np.random.randint(0, self.width - self.crop_sz_W)
            cy = np.random.randint(0, self.height - self.crop_sz_H)
        else:
            cx = int((self.width - self.crop_sz_W) / 2)
            cy = int((self.height - self.crop_sz_H) / 2)

        exp = self.exp_time
        delay = self.delay_time

        rs1_start = [index * 1000000. / self.fps[self.folder - 1] + delay * i for i in range(H)]
        rs1_end = [index * 1000000. / self.fps[self.folder - 1] + delay * i + exp for i in range(H)]
        rs2_start = [(index + 1) * 1000000. / self.fps[self.folder - 1] + delay * i for i in range(H)]
        rs2_end = [(index + 1) * 1000000. / self.fps[self.folder - 1] + delay * i + exp for i in range(H)]
        # select1 = int(self.height / 2)
        # target_exp = self.in_t_us * np.random.randint(22, 72)
        target_exp = exp
        # gs_path = Path(self.GT_file[100 * index + select1 + 5])
        # gt = self.get_img_file(gs_path)
        if self.mode == 'train':
            target = rs1_start[int(H / 3)] + np.random.rand() * (rs2_start[int(H * 2 / 3)] - rs1_start[int(H / 3)])
        else:
            target = rs1_start[0] + (rs1_end[-1] - rs1_start[0]) * self.target
            # target = rs1_start[0] + (rs2_start[0] - rs1_start[0]) * self.target
            # target = rs1_start[H//2] + (rs2_start[H//2] - rs1_start[H//2]) * self.target
        target = [target for i in range(H)]
        # target = [self.in_t_us * 100 * index + self.in_t_us * select1 for i in range(H)]
        # alignline1 = 2 * (target[0] - self.in_t_us * 100 * index) / self.in_t_us
        # alignline2 = 2 * (target[0] - self.in_t_us * exp - self.in_t_us * 100 * index) / self.in_t_us

        target = np.asarray(target)
        rs1_start = np.asarray(rs1_start)
        rs1_end = np.asarray(rs1_end)
        rs2_start = np.asarray(rs2_start)
        rs2_end = np.asarray(rs2_end)
        if self.mode == 'train':
            target = target[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy # - self.in_t_us * cy
            rs1_start = rs1_start[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy
            rs1_end = rs1_end[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy
            rs2_start = rs2_start[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy
            rs2_end = rs2_end[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy
        # print(self.subev_file[index])

        x, y, p, t = filter_events_by_space(self.x, self.y, self.p, self.t, cx, cx + self.crop_sz_W)
        y, x, p, t = filter_events_by_space(y, x, p, t, cy, cy + self.crop_sz_H)
        t, x, y, p = filter_events_by_space(t, x, y, p, rs1_start[0] - 0.7 * (rs1_end[-1] - rs1_start[0]), rs2_end[-1])
        x = x - cx
        y = y - cy

        total_event = {'x': x.astype(np.int64), 'y': y.astype(np.int64), 'p': p.astype(np.int64),
                       't': t.astype(np.int64)}
        # load_ev_time = int(time.time())
        gs_to_rs1_start_frame, gs_to_rs1_end_frame = event2frame(total_event,
                                                                 imgsize=[
                                                                     self.crop_sz_H,
                                                                     self.crop_sz_W],
                                                                 low_limit=target,
                                                                 high_limit1=rs1_start,
                                                                 high_limit2=rs1_end,
                                                                 num_frame=self.num_frame,
                                                                 noise=self.noise)

        _, target_exp_frame, = event2frame(total_event, imgsize=[self.crop_sz_H,
                                                                 self.crop_sz_W],
                                           low_limit=target,
                                           high_limit1=target,
                                           high_limit2=target + target_exp,
                                           num_frame=self.num_frame,
                                           noise=self.noise)
        rs1_to_gs_start_frame, rs1_to_gs_end_frame = event2frame(total_event,
                                                                 imgsize=[
                                                                     self.crop_sz_H,
                                                                     self.crop_sz_W],
                                                                 low_limit=rs1_start,
                                                                 high_limit1=target,
                                                                 high_limit2=target + target_exp,
                                                                 num_frame=self.num_frame,
                                                                 noise=self.noise)
        _, rs1_exp_frame, = event2frame(total_event, imgsize=[self.crop_sz_H,
                                                              self.crop_sz_W],
                                        low_limit=rs1_start,
                                        high_limit1=rs1_start,
                                        high_limit2=rs1_end,
                                        num_frame=self.num_frame,
                                        noise=self.noise)
        gs_to_rs2_start_frame, gs_to_rs2_end_frame = event2frame(total_event,
                                                                 imgsize=[
                                                                     self.crop_sz_H,
                                                                     self.crop_sz_W],
                                                                 low_limit=target,
                                                                 high_limit1=rs2_start,
                                                                 high_limit2=rs2_end,
                                                                 num_frame=self.num_frame,
                                                                 noise=self.noise)

        rs2_to_gs_start_frame, rs2_to_gs_end_frame = event2frame(total_event,
                                                                 imgsize=[
                                                                     self.crop_sz_H,
                                                                     self.crop_sz_W],
                                                                 low_limit=rs2_start,
                                                                 high_limit1=target,
                                                                 high_limit2=target + target_exp,
                                                                 num_frame=self.num_frame,
                                                                 noise=self.noise)
        _, rs2_exp_frame, = event2frame(total_event, imgsize=[self.crop_sz_H,
                                                              self.crop_sz_W],
                                        low_limit=rs2_start,
                                        high_limit1=rs2_start,
                                        high_limit2=rs2_end,
                                        num_frame=self.num_frame,
                                        noise=self.noise)
        rs1 = rs1[..., cy:cy + self.crop_sz_H, cx:cx + self.crop_sz_W]
        rs2 = rs2[..., cy:cy + self.crop_sz_H, cx:cx + self.crop_sz_W]
        # gt = gs[..., cy:cy + self.crop_sz_H, cx:cx + self.crop_sz_W]
        rs1_to_rs2_start_frame, rs1_to_rs2_end_frame = event2frame(total_event,
                                                                   imgsize=[
                                                                       self.crop_sz_H,
                                                                       self.crop_sz_W],
                                                                   low_limit=rs1_start,
                                                                   high_limit1=rs2_start,
                                                                   high_limit2=rs2_end,
                                                                   num_frame=self.num_frame,
                                                                   noise=self.noise)
        rs2_to_rs1_start_frame, rs2_to_rs1_end_frame = event2frame(total_event,
                                                                   imgsize=[
                                                                       self.crop_sz_H,
                                                                       self.crop_sz_W],
                                                                   low_limit=rs2_start,
                                                                   high_limit1=rs1_start,
                                                                   high_limit2=rs1_end,
                                                                   num_frame=self.num_frame,
                                                                   noise=self.noise)
        rs2_exp_frame = np.zeros((self.num_frame, 2, self.crop_sz_H, self.crop_sz_W))
        rs2_exp_frame = e2f_detail(total_event, rs2_exp_frame, rs2_start, rs2_end, self.noise)
        rs2_exp_frame = rs2_exp_frame.reshape(2 * self.num_frame, self.crop_sz_H, self.crop_sz_W)

        out = {
            'file_index': self.RS_file[index],
            'Orig1': rs1,
            'Orig2': rs2,
            'TtoO1_sf': gs_to_rs1_start_frame,
            'TtoO1_ef': gs_to_rs1_end_frame,
            'TtoO2_sf': gs_to_rs2_start_frame,
            'TtoO2_ef': gs_to_rs2_end_frame,
            'Texpf': target_exp_frame,
            'O1toT_sf': rs1_to_gs_start_frame,
            'O1toT_ef': rs1_to_gs_end_frame,
            'O1expf': rs1_exp_frame,
            # 'start_line': cy
            'O2toT_sf': rs2_to_gs_start_frame,
            'O2toT_ef': rs2_to_gs_end_frame,
            'O2expf': rs2_exp_frame,
            'O1toO2_sf': rs1_to_rs2_start_frame,
            'O1toO2_ef': rs1_to_rs2_end_frame,
            'O2toO1_sf': rs2_to_rs1_start_frame,
            'O2toO1_ef': rs2_to_rs1_end_frame

        }
        return out


class sequence_fastec(Dataset):
    def __init__(self, seq_path: Path, mode: str = 'train', bei=0.5,num_bins=16, crop_sz_W=256,
                 crop_sz_H=256, noise=0.1):
        assert num_bins >= 1
        assert seq_path.is_dir()
        self.seq_path = seq_path
        # NOTE: Adapt this code according to the present mode (e.g. train, val or test).
        self.mode = mode
        self.height = 480
        self.width = 640
        self.bei =  bei
        if self.mode == 'train':

            self.crop_sz_W = crop_sz_W
            self.crop_sz_H = crop_sz_H
        else:
            self.crop_sz_W = self.width
            self.crop_sz_H = self.height

        self.num_bins = num_bins
        self.num_frame = num_bins
        # self.num_frame
        self.noise = noise
        self.frame_time = np.loadtxt(seq_path / 'new' / 'dvs-video-frame_times.txt')
        # self.target = target
        # Save interval timestamp in ms
        self.in_t_us = 1 / 2400
        self.fps = 2400
        # load RS  image
        RS_dir = seq_path
        assert RS_dir.is_dir()
        RS_file = list()
        for entry in RS_dir.iterdir():
            if 'rolling' in str(entry):
                RS_file.append(str(entry))
        RS_file.sort()
        self.RS_file = RS_file
        # load event
        subev_dir = seq_path / 'newev'
        assert subev_dir.is_dir()
        subev_file = list()
        for entry in subev_dir.iterdir():
            if str(entry.name).endswith('npz'):
                subev_file.append(str(entry))
        subev_file.sort()
        self.subev_file = subev_file

        # load GT
        GT_dir = seq_path
        assert GT_dir.is_dir()
        GT_file = list()
        for entry in GT_dir.iterdir():
            if 'global' in str(entry):
                GT_file.append(str(entry))
        GT_file.sort()
        self.GT_file = GT_file

    @staticmethod
    def get_img_file(filepath: Path):
        assert filepath.is_file()
        img = Image.open(str(filepath))
        img = np.asarray(img)
        img = F.to_tensor(img)
        return img

    def __len__(self):
        return len(self.RS_file)- 1

    def __getitem__(self, index):

        # start_time = int(time.time())

        RS_path = Path(self.RS_file[index])
        rs1 = self.get_img_file(RS_path)
        RS_path = Path(self.RS_file[index + 1])
        rs2 = self.get_img_file(RS_path)
        C, H, W = rs1.shape

        if self.mode == 'train':
            cx = np.random.randint(0, self.width - self.crop_sz_W)
            cy = np.random.randint(0, self.height - self.crop_sz_H)
        else:
            cx = int((self.width - self.crop_sz_W) / 2)
            cy = int((self.height - self.crop_sz_H) / 2)

        exp = 16
        delay = 240
        rs1_start = [self.in_t_us * 240 * index + self.in_t_us * i / 2 for i in range(2*H)]
        rs1_end = [exp * self.in_t_us + self.in_t_us * 240 * index + self.in_t_us * i / 2 for i in range(H)]
        rs2_start = [self.in_t_us * 240 * (index + 1) + self.in_t_us * i / 2 for i in range(H)]
        rs2_end = [exp * self.in_t_us + self.in_t_us * 240 * (index + 1) + self.in_t_us * i / 2 for i in
                   range(H)]
        # select1 = np.random.randint(180, 280)
        if self.mode == 'train':
            select1 = H - 1
            gs_path = Path(self.GT_file[2 * index + 2])
        else:
            # select1 = int(H / 2)
            select1 = int(self.bei*H)
            gs_path = Path(self.GT_file[2 * index + 1])
            # select1 = H - 1
            # gs_path = Path(self.GT_file[2 * index + 2])
        # target_exp = self.in_t_us * np.random.randint(44, 144)
        # target_exp = self.in_t_us * 4
        # gs_path = Path(self.GT_file[100 * index + select1 + 2])
        target_exp = self.in_t_us * 16
        #
        gs = self.get_img_file(gs_path)
        # target = [self.in_t_us * 100 * index + self.in_t_us * select1 for i in range(H)]
        # target = [self.in_t_us * 100 * index + self.in_t_us * i / 2 + self.in_t_us * exp/2 for i in range(H)]
        target = [rs1_start[select1] for i in range(H)]
        # target1 = [rs1_start[select1 + 10] for i in range(H)]
        # alignline1 = (target[0] - self.in_t_us * 100 * index) / self.in_t_us
        # alignline2 = (target[0] - self.in_t_us * exp - self.in_t_us * 100 * index) / self.in_t_us

        target = np.asarray(target)

        rs1_start = np.asarray(rs1_start)
        rs1_end = np.asarray(rs1_end)
        rs2_start = np.asarray(rs2_start)
        rs2_end = np.asarray(rs2_end)
        # if self.mode == 'train':
        target = target[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy # - self.in_t_us * cy

        rs1_start = rs1_start[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy
        rs1_end = rs1_end[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy
        rs2_start = rs2_start[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy
        rs2_end = rs2_end[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy
        # print(self.subev_file[index])
        event1 = np.load(self.subev_file[index])
        x1, y1, p1, t1 = event1['x'], event1['y'], event1['p'], event1['t']
        x1 = x1.astype(np.int64)
        y1 = y1.astype(np.int64)
        p1 = p1.astype(np.int64)
        event2 = np.load(self.subev_file[index + 1])
        x2, y2, p2, t2 = event2['x'], event2['y'], event2['p'], event2['t']
        x2 = x2.astype(np.int64)
        y2 = y2.astype(np.int64)
        p2 = p2.astype(np.int64)
        # t1_, y1_, p1_, x1_ = filter_events_by_space(t1, y1, p1, x1,
        #                                             self.in_t_us * 100 * index + self.in_t_us * 100,
        #                                             self.in_t_us * 100 * index + self.in_t_us * delay + self.in_t_us * exp)

        # t2, y2, p2, x2 = filter_events_by_space(t2, y2, p2, x2,
        #                                         rs1_end[-1],
        #                                         100000000)
        x = np.concatenate((x1, x2), 0)
        y = np.concatenate((y1, y2), 0)
        p = np.concatenate((p1, p2), 0)
        t = np.concatenate((t1, t2), 0)
        # if self.mode == 'train':
        x, y, p, t = filter_events_by_space(x, y, p, t, cx, cx + self.crop_sz_W)
        y, x, p, t = filter_events_by_space(y, x, p, t, cy, cy + self.crop_sz_H)
        x = x - cx
        y = y - cy

        total_event = {'x': x, 'y': y, 'p': p, 't': t}
        # load_ev_time = int(time.time())
        gs_to_rs1_start_frame, gs_to_rs1_end_frame = event2frame(total_event,
                                                                 imgsize=[
                                                                     self.crop_sz_H,
                                                                     self.crop_sz_W],
                                                                 low_limit=target,
                                                                 high_limit1=rs1_start,
                                                                 high_limit2=rs1_end,
                                                                 num_frame=self.num_frame,
                                                                 noise=self.noise)
        target_exp_frame = np.zeros((self.num_frame, 2, self.crop_sz_H, self.crop_sz_W))
        target_exp_frame = e2f_detail(total_event, target_exp_frame, target, target + target_exp, self.noise)
        target_exp_frame = target_exp_frame.reshape(2 * self.num_frame, self.crop_sz_H, self.crop_sz_W)
        rs1_to_gs_start_frame, rs1_to_gs_end_frame = event2frame(total_event,
                                                                 imgsize=[
                                                                     self.crop_sz_H,
                                                                     self.crop_sz_W],
                                                                 low_limit=rs1_start,
                                                                 high_limit1=target,
                                                                 high_limit2=target + target_exp,
                                                                 num_frame=self.num_frame,
                                                                 noise=self.noise)
        rs1_exp_frame = np.zeros((self.num_frame, 2, self.crop_sz_H, self.crop_sz_W))
        rs1_exp_frame = e2f_detail(total_event, rs1_exp_frame, rs1_start, rs1_end, self.noise)
        rs1_exp_frame = rs1_exp_frame.reshape(2 * self.num_frame, self.crop_sz_H, self.crop_sz_W)
        gs_to_rs2_start_frame, gs_to_rs2_end_frame = event2frame(total_event,
                                                                 imgsize=[
                                                                     self.crop_sz_H,
                                                                     self.crop_sz_W],
                                                                 low_limit=target,
                                                                 high_limit1=rs2_start,
                                                                 high_limit2=rs2_end,
                                                                 num_frame=self.num_frame,
                                                                 noise=self.noise)

        rs2_to_gs_start_frame, rs2_to_gs_end_frame = event2frame(total_event,
                                                                 imgsize=[
                                                                     self.crop_sz_H,
                                                                     self.crop_sz_W],
                                                                 low_limit=rs2_start,
                                                                 high_limit1=target,
                                                                 high_limit2=target + target_exp,
                                                                 num_frame=self.num_frame,
                                                                 noise=self.noise)
        rs1_to_rs2_start_frame, rs1_to_rs2_end_frame = event2frame(total_event,
                                                                   imgsize=[
                                                                       self.crop_sz_H,
                                                                       self.crop_sz_W],
                                                                   low_limit=rs1_start,
                                                                   high_limit1=rs2_start,
                                                                   high_limit2=rs2_end,
                                                                   num_frame=self.num_frame,
                                                                   noise=self.noise)
        rs2_to_rs1_start_frame, rs2_to_rs1_end_frame = event2frame(total_event,
                                                                   imgsize=[
                                                                       self.crop_sz_H,
                                                                       self.crop_sz_W],
                                                                   low_limit=rs2_start,
                                                                   high_limit1=rs1_start,
                                                                   high_limit2=rs1_end,
                                                                   num_frame=self.num_frame,
                                                                   noise=self.noise)
        rs2_exp_frame = np.zeros((self.num_frame, 2, self.crop_sz_H, self.crop_sz_W))
        rs2_exp_frame = e2f_detail(total_event, rs2_exp_frame, rs2_start, rs2_end, self.noise)
        rs2_exp_frame = rs2_exp_frame.reshape(2 * self.num_frame, self.crop_sz_H, self.crop_sz_W)

        rs1 = rs1[..., cy:cy + self.crop_sz_H, cx:cx + self.crop_sz_W]
        rs2 = rs2[..., cy:cy + self.crop_sz_H, cx:cx + self.crop_sz_W]
        gt = gs[..., cy:cy + self.crop_sz_H, cx:cx + self.crop_sz_W]
        select = np.random.randint(0, len(self.GT_file))
        # gs_path = Path(self.GT_file[select])
        # gs = self.get_img_file(gs_path)
        # gs = gs[..., cy:cy + self.crop_sz_H, cx:cx + self.crop_sz_W]

        out = {
            'file_index': self.RS_file[index],
            'Orig1': rs1,
            'Orig2': rs2,
            'TtoO1_sf': gs_to_rs1_start_frame,
            'TtoO1_ef': gs_to_rs1_end_frame,
            'TtoO2_sf': gs_to_rs2_start_frame,
            'TtoO2_ef': gs_to_rs2_end_frame,
            'Texpf': target_exp_frame,
            'O1toT_sf': rs1_to_gs_start_frame,
            'O1toT_ef': rs1_to_gs_end_frame,
            'O1expf': rs1_exp_frame,
            'O2toT_sf': rs2_to_gs_start_frame,
            'O2toT_ef': rs2_to_gs_end_frame,
            'O2expf': rs2_exp_frame,
            'O1toO2_sf': rs1_to_rs2_start_frame,
            'O1toO2_ef': rs1_to_rs2_end_frame,

            'O2toO1_sf': rs2_to_rs1_start_frame,
            'O2toO1_ef': rs2_to_rs1_end_frame,
            'GT': gt,
            # 'gs': gs
        }
        return out

class sequence_dre(Dataset):
    def __init__(self, seq_path: Path, target=0.5, mode: str = 'train', num_bins=16, crop_sz_W=256,
                 crop_sz_H=256, noise=0.1):
        assert num_bins >= 1
        assert seq_path.is_dir()
        self.seq_path = seq_path
        self.mode = mode
        self.target = target
        self.reverse = 0
        # Save output dimensions
        self.height = 360
        self.width = 640
        self.num_bins = num_bins
        self.num_frame = num_bins
        self.noise = noise
        # self.target = target
        # Save interval timestamp in ms
        self.in_t_us = 1 / 13171.
        # load RS  image

        RS_dir = seq_path / 'images'
        assert RS_dir.is_dir()
        RS_file = list()
        for entry in RS_dir.iterdir():
            assert str(entry.name).endswith('.png')
            RS_file.append(str(entry))
        RS_file.sort()
        self.RS_file = RS_file
        self.height = 260
        self.width = 346

        if self.mode == 'train':
            self.crop_sz_W = crop_sz_W
            self.crop_sz_H = crop_sz_H
        else:
            self.crop_sz_W = self.width
            self.crop_sz_H = self.height
        for entry in self.seq_path.iterdir():
            if str(entry.name).endswith('h5'):
                self.ev_file = str(entry)
        with h5py.File(self.ev_file) as f:
            self.img_ts = np.asarray(f['image_ts'])
            self.start_img_ts = np.asarray(f['start_image_ts'])
            self.end_img_ts = np.asarray(f['end_image_ts'])
            self.x = np.asarray(f['x'])
            self.y = np.asarray(f['y'])
            self.p = np.asarray(f['p'])
            self.t = np.asarray(f['t'])
            self.x, self.y, self.p, self.t = self.x.astype(np.int64), self.y.astype(np.int64), self.p.astype(np.int64), self.t.astype(np.int64)

            self.end_img_ts = self.end_img_ts - self.start_img_ts.min()
            self.t = self.t - self.start_img_ts.min()
            self.start_img_ts = self.start_img_ts - self.start_img_ts.min()
        self.p[self.p == -1] = 0
        self.exp_time = self.end_img_ts - self.start_img_ts
        self.delay_time = 70

    @staticmethod
    def get_img_file(filepath: Path):

        assert filepath.is_file()
        # img_input = cv2.cvtColor(cv2.imread(str(filepath)), cv2.COLOR_BGR2RGB)
        # img_input = torch.from_numpy(img_input.copy()).permute(2, 0, 1).float() / 255
        img_input = cv2.imread(str(filepath))
        img_input = torch.from_numpy(img_input.copy()).permute(2, 0, 1).float() / 255
        return img_input

    def __len__(self):
        return len(self.RS_file) - 1

    def __getitem__(self, index):
        # index+=38
        RS_path = Path(self.RS_file[index])
        rs1 = self.get_img_file(RS_path)
        RS_path = Path(self.RS_file[index + 1])
        rs2 = self.get_img_file(RS_path)
        C, H, W = rs1.shape
        # self.height = H
        # self.width = W
        if self.mode == 'train':
            cx = np.random.randint(0, self.width - self.crop_sz_W)
            cy = np.random.randint(0, self.height - self.crop_sz_H)
        else:
            cx = int((self.width - self.crop_sz_W) / 2)
            cy = int((self.height - self.crop_sz_H) / 2)

        exp = self.exp_time[index]
        delay = self.delay_time

        rs1_start = [self.start_img_ts[index] + delay * i for i in range(H)]
        rs1_end = [self.end_img_ts[index] + delay * i  for i in range(H)]
        rs2_start = [self.start_img_ts[index+1] + delay * i for i in range(H)]
        rs2_end = [self.end_img_ts[index+1] + delay * i for i in range(H)]
        # select1 = int(self.height / 2)
        # target_exp = self.in_t_us * np.random.randint(22, 72)
        target_exp = exp
        # gs_path = Path(self.GT_file[100 * index + select1 + 5])
        # gt = self.get_img_file(gs_path)
        if self.mode == 'train':
            target = rs1_start[int(H / 3)] + np.random.rand() * (rs2_start[int(H * 2 / 3)] - rs1_start[int(H / 3)])
        else:
            target = rs1_start[0] + (rs1_end[-1] - rs1_start[0]) * self.target
            # target = rs2_start[0] + (rs2_end[-1] - rs2_start[0]) * self.target
            # target = rs1_start[H // 2] + (rs2_start[H // 2] - rs1_start[H // 2]) * self.target
        target = [target for i in range(H)]
        # target = [self.in_t_us * 100 * index + self.in_t_us * select1 for i in range(H)]
        # alignline1 = 2 * (target[0] - self.in_t_us * 100 * index) / self.in_t_us
        # alignline2 = 2 * (target[0] - self.in_t_us * exp - self.in_t_us * 100 * index) / self.in_t_us

        target = np.asarray(target)
        rs1_start = np.asarray(rs1_start)
        rs1_end = np.asarray(rs1_end)
        rs2_start = np.asarray(rs2_start)
        rs2_end = np.asarray(rs2_end)
        if self.mode == 'train':
            target = target[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy # - self.in_t_us * cy
            rs1_start = rs1_start[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy
            rs1_end = rs1_end[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy
            rs2_start = rs2_start[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy
            rs2_end = rs2_end[cy:cy + self.crop_sz_H]  # - self.in_t_us * cy
        # print(self.subev_file[index])

        x, y, p, t = filter_events_by_space(self.x, self.y, self.p, self.t, cx, cx + self.crop_sz_W)
        y, x, p, t = filter_events_by_space(y, x, p, t, cy, cy + self.crop_sz_H)
        t, x, y, p = filter_events_by_space(t, x, y, p, rs1_start[0], rs2_end[-1])
        x = x - cx
        y = y - cy

        total_event = {'x': x.astype(np.int64), 'y': y.astype(np.int64), 'p': p.astype(np.int64),
                       't': t.astype(np.int64)}
        # load_ev_time = int(time.time())
        gs_to_rs1_start_frame, gs_to_rs1_end_frame = event2frame(total_event,
                                                                 imgsize=[
                                                                     self.crop_sz_H,
                                                                     self.crop_sz_W],
                                                                 low_limit=target,
                                                                 high_limit1=rs1_start,
                                                                 high_limit2=rs1_end,
                                                                 num_frame=self.num_frame,
                                                                 noise=self.noise)

        _, target_exp_frame, = event2frame(total_event, imgsize=[self.crop_sz_H,
                                                                 self.crop_sz_W],
                                           low_limit=target,
                                           high_limit1=target,
                                           high_limit2=target + target_exp,
                                           num_frame=self.num_frame,
                                           noise=self.noise)
        rs1_to_gs_start_frame, rs1_to_gs_end_frame = event2frame(total_event,
                                                                 imgsize=[
                                                                     self.crop_sz_H,
                                                                     self.crop_sz_W],
                                                                 low_limit=rs1_start,
                                                                 high_limit1=target,
                                                                 high_limit2=target + target_exp,
                                                                 num_frame=self.num_frame,
                                                                 noise=self.noise)
        _, rs1_exp_frame, = event2frame(total_event, imgsize=[self.crop_sz_H,
                                                              self.crop_sz_W],
                                        low_limit=rs1_start,
                                        high_limit1=rs1_start,
                                        high_limit2=rs1_end,
                                        num_frame=self.num_frame,
                                        noise=self.noise)
        gs_to_rs2_start_frame, gs_to_rs2_end_frame = event2frame(total_event,
                                                                 imgsize=[
                                                                     self.crop_sz_H,
                                                                     self.crop_sz_W],
                                                                 low_limit=target,
                                                                 high_limit1=rs2_start,
                                                                 high_limit2=rs2_end,
                                                                 num_frame=self.num_frame,
                                                                 noise=self.noise)

        rs2_to_gs_start_frame, rs2_to_gs_end_frame = event2frame(total_event,
                                                                 imgsize=[
                                                                     self.crop_sz_H,
                                                                     self.crop_sz_W],
                                                                 low_limit=rs2_start,
                                                                 high_limit1=target,
                                                                 high_limit2=target + target_exp,
                                                                 num_frame=self.num_frame,
                                                                 noise=self.noise)
        _, rs2_exp_frame, = event2frame(total_event, imgsize=[self.crop_sz_H,
                                                              self.crop_sz_W],
                                        low_limit=rs2_start,
                                        high_limit1=rs2_start,
                                        high_limit2=rs2_end,
                                        num_frame=self.num_frame,
                                        noise=self.noise)
        rs1 = rs1[..., cy:cy + self.crop_sz_H, cx:cx + self.crop_sz_W]
        rs2 = rs2[..., cy:cy + self.crop_sz_H, cx:cx + self.crop_sz_W]
        # gt = gs[..., cy:cy + self.crop_sz_H, cx:cx + self.crop_sz_W]
        rs1_to_rs2_start_frame, rs1_to_rs2_end_frame = event2frame(total_event,
                                                                   imgsize=[
                                                                       self.crop_sz_H,
                                                                       self.crop_sz_W],
                                                                   low_limit=rs1_start,
                                                                   high_limit1=rs2_start,
                                                                   high_limit2=rs2_end,
                                                                   num_frame=self.num_frame,
                                                                   noise=self.noise)
        rs2_to_rs1_start_frame, rs2_to_rs1_end_frame = event2frame(total_event,
                                                                   imgsize=[
                                                                       self.crop_sz_H,
                                                                       self.crop_sz_W],
                                                                   low_limit=rs2_start,
                                                                   high_limit1=rs1_start,
                                                                   high_limit2=rs1_end,
                                                                   num_frame=self.num_frame,
                                                                   noise=self.noise)
        rs2_exp_frame = np.zeros((self.num_frame, 2, self.crop_sz_H, self.crop_sz_W))
        rs2_exp_frame = e2f_detail(total_event, rs2_exp_frame, rs2_start, rs2_end, self.noise)
        rs2_exp_frame = rs2_exp_frame.reshape(2 * self.num_frame, self.crop_sz_H, self.crop_sz_W)

        out = {
            'file_index': self.RS_file[index],
            'Orig1': rs1,
            'Orig2': rs2,
            'TtoO1_sf': gs_to_rs1_start_frame,
            'TtoO1_ef': gs_to_rs1_end_frame,
            'TtoO2_sf': gs_to_rs2_start_frame,
            'TtoO2_ef': gs_to_rs2_end_frame,
            'Texpf': target_exp_frame,
            'O1toT_sf': rs1_to_gs_start_frame,
            'O1toT_ef': rs1_to_gs_end_frame,
            'O1expf': rs1_exp_frame,
            # 'start_line': cy
            'O2toT_sf': rs2_to_gs_start_frame,
            'O2toT_ef': rs2_to_gs_end_frame,
            'O2expf': rs2_exp_frame,
            'O1toO2_sf': rs1_to_rs2_start_frame,
            'O1toO2_ef': rs1_to_rs2_end_frame,
            'O2toO1_sf': rs2_to_rs1_start_frame,
            'O2toO1_ef': rs2_to_rs1_end_frame

        }
        return out