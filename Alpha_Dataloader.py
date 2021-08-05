#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   Alpha_Dataloader.py
@Author :   Song
@Time   :   2020/7/21 12:32
@Contact:   songjian@westlake.edu.cn
@intro  : 
'''
import torch
from torch.utils.data.dataset import Dataset


class Alpha_Dataset(Dataset):
    def __init__(self, X, X_pr_charge, X_seq_len, y):
        print(f'Train, df_input: {len(y)}')

        # to numpy
        self.X = X
        self.X_pr_charge = X_pr_charge
        self.X_seq_len = X_seq_len
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        tri = self.X[idx]
        tri = tri.reshape(3, -1)
        tri = tri.T

        pr_charge = self.X_pr_charge[idx]
        seq_len = self.X_seq_len[idx]

        # y
        y = self.y[idx]
        return (torch.tensor(tri), pr_charge, seq_len, y)
