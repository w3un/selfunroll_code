import torch
from pathlib import Path



from dataloader.sequence import sequence_gevreal,sequence_gev,sequence_fastec,sequence_dre


class GevLoader:
    def __init__(self, dataset_path: str, target=0.5,num_bins=16, crop_sz_W=128, crop_sz_H=128):
        dataset_path = Path(dataset_path)
        train_path = dataset_path / 'train'
        test_path = dataset_path / 'test'
        assert dataset_path.is_dir(), str(dataset_path)
        assert train_path.is_dir(), str(train_path)

        train_sequences = list()
        test_sequences = list()
        for child in train_path.iterdir():
            train_sequences.append(
                sequence_gev(child, 'train', target,num_bins, crop_sz_H=crop_sz_H, crop_sz_W=crop_sz_W, noise=0))
        for child in test_path.iterdir():
            test_sequences.append(
                sequence_gev(child, 'test', target,num_bins, crop_sz_H=crop_sz_H, crop_sz_W=crop_sz_W, noise=0))
        self.train_dataset = torch.utils.data.ConcatDataset(train_sequences)
        self.test_dataset = torch.utils.data.ConcatDataset(test_sequences)

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset
class GevrealLoader:
    def __init__(self, dataset_path: str, target=0.5, num_bins=16, crop_sz_W=128, crop_sz_H=128):
        dataset_path = Path(dataset_path)
        train_path = dataset_path / 'train'
        val_path = dataset_path / 'validate'
        test_path = dataset_path / 'test'
        assert dataset_path.is_dir(), str(dataset_path)
        assert train_path.is_dir(), str(train_path)

        train_sequences = list()
        val_sequences = list()
        test_sequences = list()
        # for child in train_path.iterdir():

        for child in train_path.iterdir():
            train_sequences.append(
                sequence_gevreal(child, target, 'train', num_bins, crop_sz_H=crop_sz_H, crop_sz_W=crop_sz_W, noise=0))
        for child in test_path.iterdir():
            test_sequences.append(
                sequence_gevreal(child,  target,'test', num_bins, crop_sz_H=crop_sz_H, crop_sz_W=crop_sz_W,
                                 noise=0))
        self.train_dataset = torch.utils.data.ConcatDataset(train_sequences)
        self.test_dataset = torch.utils.data.ConcatDataset(test_sequences)

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset
class FastecLoader:
    def __init__(self, dataset_path: str, bei = 0.5, num_bins=16, crop_sz_W=128, crop_sz_H=128):
        dataset_path = Path(dataset_path)
        train_path = dataset_path / 'train'
        val_path = dataset_path / 'validate'
        test_path = dataset_path / 'test'
        assert dataset_path.is_dir(), str(dataset_path)
        assert train_path.is_dir(), str(train_path)

        train_sequences = list()
        test_sequences = list()

        for child in train_path.iterdir():
            train_sequences.append(
                sequence_fastec(child, 'train', bei,num_bins, crop_sz_H=crop_sz_H, crop_sz_W=crop_sz_W, noise=0))
        for child in test_path.iterdir():
            test_sequences.append(
                sequence_fastec(child, 'test',bei, num_bins, crop_sz_H=crop_sz_H, crop_sz_W=crop_sz_W, noise=0))
        self.train_dataset = torch.utils.data.ConcatDataset(train_sequences)
        self.test_dataset = torch.utils.data.ConcatDataset(test_sequences)

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

class DrerealLoader:
    def __init__(self, dataset_path: str, target=0.5 , num_bins=16, crop_sz_H=128, crop_sz_W=128):
        dataset_path = Path(dataset_path)
        self.train_path = dataset_path / 'Train'
        self.test_path = dataset_path / 'Test'

        self.target = target
        self.num_bins = num_bins
        self.crop_sz_H = crop_sz_H
        self.crop_sz_W = crop_sz_W
        self.train_sequences = list()
        self.test_sequences = list()

        

    def get_train_dataset(self):
        for child in self.train_path.iterdir():
            self.train_sequences.append(
                sequence_dre(child, self.target, 'train',self.num_bins, crop_sz_H=self.crop_sz_H, crop_sz_W=self.crop_sz_W, noise=0))
        self.train_dataset = torch.utils.data.ConcatDataset(self.train_sequences)
        return self.train_dataset

    

    def get_test_dataset(self):
        for child in self.test_path.iterdir():
            self.test_sequences.append(
                sequence_dre(child, self.target, 'test', self.num_bins,crop_sz_H=self.crop_sz_H, crop_sz_W=self.crop_sz_W, noise=0))
        self.test_dataset = torch.utils.data.ConcatDataset(self.test_sequences)
        return self.test_dataset

import os
import time
import argparse
import gc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--is_flip_rotate', type=bool, default=False, help='flag used for data augumentation')
    parser.add_argument('--crop_sz_H', type=int, default=144, help='cropped image size height')
    parser.add_argument('--crop_sz_W', type=int, default=144, help='cropped image size width')
    parser.add_argument('--datapath', type=str, default='/home/wyg/Documents/wyg/fastec_rs_train/')
    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--batch_sz', type=int, default=16, help='batch size used for training')
    parser.add_argument('--num_bins', type=int, default=8, help='')
    parser.add_argument('--num_frames', type=int, default=32, help='')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    opts = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    dataset_provider = FastecLoader(opts.datapath, opts.num_bins)
    train_data = dataset_provider.get_train_dataset()
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=opts.batch_sz, shuffle=False,
                                               num_workers=opts.num_workers, pin_memory=False, drop_last=True)
    # last_time = time()
    for cuiter, sample in enumerate(train_loader):
        print(cuiter)
        # if cuiter % 10 == 0:

        # optimizer.zero_grad()
        # rs_blur = sample['rs_blur'].cuda().float()
        # rs_ev = sample['rs_evxol'].cuda().float()
