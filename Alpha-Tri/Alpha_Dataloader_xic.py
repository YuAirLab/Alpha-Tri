#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :
@Author :   Song
@Time   :   2021/1/11 15:33
@Contact:   songjian@westlake.edu.cn
@intro  : 
'''
import numpy as np
from torch.utils.data.dataset import Dataset

try:
    profile
except:
    profile = lambda x: x

class Model_Dataset_xic(Dataset):
    def __init__(self, xics, xics_len, y, type):
        if type != 'test':
            print(f'****** {type} numbers of XICs: {len(y)}')

        self.type = type

        # to numpy
        self.xics = xics
        self.xics_len = xics_len

        self.y = y

        self.row_idx = [0]
        self.row_idx.extend(xics_len)
        self.row_idx = np.array(self.row_idx).cumsum()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        idx_start = self.row_idx[idx]
        idx_end = self.row_idx[idx+1]

        xic = self.xics[idx_start : idx_end]
        y = self.y[idx]

        # shuffle
        if self.type == 'train':
            np.random.shuffle(xic)

        return (xic, xic.shape[0], y)