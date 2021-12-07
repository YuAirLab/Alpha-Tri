#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   Model_GRU_tri.py
@Author :   Song
@Time   :   2020/3/19 14:54
@Contact:   songjian@westlake.edu.cn
@intro  :
'''
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class Model_GRU_tri(nn.Module):

    def __init__(self):
        super(Model_GRU_tri, self).__init__()
        lstm_out_dim = 256
        embed = 32

        self.len_embed = nn.Embedding(30, embed)
        self.charge_embed = nn.Embedding(4, embed)

        # seq
        self.gru = nn.GRU(batch_first=True,
                          bidirectional=True,
                          num_layers=2,
                          input_size=3,
                          hidden_size=int(lstm_out_dim / 2))

        # attention
        self.attention = nn.Linear(lstm_out_dim, 32)
        self.context = nn.Linear(32, 1, bias=False)

        # fc for classify
        total_feature = lstm_out_dim + embed + embed
        self.fc = nn.Linear(total_feature, 2)

    def forward(self, batch_tri, batch_tri_len, batch_pr_charge, batch_seq_len):
        # triple-spectrum
        outputs = pack_padded_sequence(batch_tri, batch_tri_len, batch_first=True, enforce_sorted=False)
        self.gru.flatten_parameters()
        outputs, _ = self.gru(outputs, None)

        # fc attention
        att_w = torch.tanh(self.attention(outputs.data))  # [batch_size, batch_lens, att_size]
        att_w = self.context(att_w).squeeze(1)  # [batch_size, batch_lens]
        max_w = att_w.max()
        att_w = torch.exp(att_w - max_w)
        att_w, _ = pad_packed_sequence(PackedSequence(data=att_w,
                                                      batch_sizes=outputs.batch_sizes,
                                                      sorted_indices=outputs.sorted_indices,
                                                      unsorted_indices=outputs.unsorted_indices), batch_first=True)
        alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)  # [batch_size, max_lens]
        # outputs
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        # weight
        outputs = (outputs * alphas.unsqueeze(2)).sum(dim=1)  # [batch_size, out_dim]

        # cat charge embed
        embed_charge = self.charge_embed(batch_pr_charge - 1)
        embed_len = self.len_embed(batch_seq_len - 1)
        result = torch.cat((outputs, embed_charge, embed_len), dim=1)

        result = self.fc(result)

        return result
