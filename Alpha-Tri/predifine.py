#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   predifine.py
@Author :   Song
@Time   :   2021/1/11 16:48
@Contact:   songjian@westlake.edu.cn
@intro  : 
'''
import torch

g_aa_to_mass = {'A': 89.0476792233, 'C': 178.04121404900002, 'D': 133.0375092233, 'E': 147.05315928710002,
                'F': 165.07897935090006, 'G': 75.0320291595, 'H': 155.06947728710003, 'I': 131.0946294147,
                'K': 146.10552844660003, 'L': 131.0946294147, 'M': 149.05105008089998, 'm': 165.04596508089998,
                'N': 132.0534932552, 'P': 115.06332928709999, 'Q': 146.06914331900003, 'R': 174.11167644660003,
                'S': 105.0425942233, 'T': 119.05824428710001, 'V': 117.0789793509, 'W': 204.0898783828,
                'Y': 181.07389435090005,
                'x': 0}

device = torch.device('cuda')

MassCorrectionMs1 = []
MassCalCenterMs1 = []
MassCalBinsMs1 = 1
MassAccuracyMs1 = 20.

MassCorrection = []
MassCalCenter = []
MassCalBins = 1
MassAccuracy = 20.

target_dim = 32
extend_time = 0.
