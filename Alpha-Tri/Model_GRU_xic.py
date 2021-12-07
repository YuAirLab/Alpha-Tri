#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   AlphaXIC_GRU.py
@Author :   Song
@Time   :   2020/3/19 14:54
@Contact:   songjian@westlake.edu.cn
@intro  :
'''
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import predifine

class Model_GRU_xic(nn.Module):

    def __init__(self):
        super(Model_GRU_xic, self).__init__()
        lstm_out_dim = 128
        att_size = 128
        # seq
        self.xic_gru = nn.GRU(batch_first=True,
                              bidirectional=True,
                              num_layers=2,
                              input_size=predifine.target_dim,
                              hidden_size=int(lstm_out_dim/2),
                              dropout=0.5)
        # attention
        self.attention = nn.Linear(lstm_out_dim, att_size)
        self.context = nn.Linear(att_size, 1, bias=False)

        # fc for classify
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        self.fc = nn.Linear(lstm_out_dim, 2)

    def forward(self, batch_xic, batch_xic_num):
        # xic
        self.xic_gru.flatten_parameters()
        batch_xic = pack_padded_sequence(batch_xic, batch_xic_num, batch_first=True, enforce_sorted=False)
        outputs, _ = self.xic_gru(batch_xic)

        '''attention'''
        att_w = torch.tanh(self.attention(outputs.data))  # [batch_size, batch_lens, att_size]
        att_w = self.context(att_w).squeeze(1)  # [batch_size*batch_lens]
        max_w = att_w.max()
        att_w = torch.exp(att_w - max_w)
        att_w, _ = pad_packed_sequence(PackedSequence(data=att_w,
                                                      batch_sizes=outputs.batch_sizes,
                                                      sorted_indices=outputs.sorted_indices,
                                                      unsorted_indices=outputs.unsorted_indices), batch_first=True)
        alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)  # [batch_size, max_lens]
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        outputs = (outputs * alphas.unsqueeze(2)).sum(dim=1)  # [batch_size, out_dim]

        # norm
        outputs = self.layer_norm(outputs)

        # fc
        result = self.fc(outputs)

        return result
