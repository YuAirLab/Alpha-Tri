#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   train.py
@Author :   Song
@Time   :   2021/1/11 19:26
@Contact:   songjian@westlake.edu.cn
@intro  : 
'''
import time
import operator
import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.utils.data
import torch.nn.functional

from sklearn.model_selection import train_test_split
from Alpha_Dataloader_xic import Model_Dataset_xic
import predifine
from Model_GRU_xic import Model_GRU_xic
from scipy.signal import savgol_filter
from sklearn import preprocessing

import matplotlib.pyplot as plt
from numba import jit

from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

try:
    profile
except:
    profile = lambda x: x


def my_collate(item):
    xic_l, xic_num, label = zip(*item)

    max_row = max(xic_num)
    xic = [np.vstack([xics, np.zeros((max_row - len(xics), xics.shape[1]))]) for xics in xic_l]

    xic = torch.tensor(xic)
    xic_num = torch.tensor(xic_num)
    label = torch.tensor(label)

    return xic, xic_num, label


def train_one_epoch(trainloader, model, optimizer, loss_fn):
    model.train()
    epoch_loss_fg = 0.
    device = predifine.device
    for batch_idx, (batch_xic, batch_xic_num, batch_labels) in enumerate(trainloader):
        batch_xic = batch_xic.float().to(device)
        batch_xic_num = batch_xic_num.long().to(device)
        batch_labels = batch_labels.long().to(device)

        # forward
        prob = model(batch_xic, batch_xic_num)

        # loss
        batch_loss = loss_fn(prob, batch_labels)

        # back
        optimizer.zero_grad()
        batch_loss.backward()
        # update
        optimizer.step()

        # loss
        epoch_loss_fg += (batch_loss.item() * len(batch_xic))

    epoch_loss_fg = epoch_loss_fg / (len(trainloader.dataset))
    return epoch_loss_fg


def eval_one_epoch(evalloader, model):
    model.eval()
    device = predifine.device
    prob_v = []
    label_v = []
    for batch_idx, (batch_xic, batch_fg_num, batch_labels) in enumerate(evalloader):
        batch_xic = batch_xic.float().to(device)
        batch_fg_num = batch_fg_num.long().to(device)
        batch_labels = batch_labels.long().to(device)

        # forward
        prob = model(batch_xic, batch_fg_num)
        prob = torch.softmax(prob.view(-1, 2), 1)
        probs = prob[:, 1].tolist()

        prob_v.extend(probs)
        label_v.extend(batch_labels.cpu().tolist())
    prob_v = np.array(prob_v)
    return prob_v, label_v


def stack_filter_xic(xics):
    xics_len = list(map(len, xics))
    xics = np.vstack(xics)
    xics = savgol_filter(xics, 11, 3, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        xics = xics / xics.max(axis=1, keepdims=True)
    xics[np.isnan(xics)] = 0

    return xics, xics_len


def train_model(X, y, random_state):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=1 / 10., random_state=random_state)
    train_xics, train_xics_len = stack_filter_xic(X_train)
    valid_xics, valid_xics_len = stack_filter_xic(X_valid)

    train_dataset = Model_Dataset_xic(train_xics, train_xics_len, y_train, type='train')
    valid_dataset = Model_Dataset_xic(valid_xics, valid_xics_len, y_valid, type='valid')

    # dataloader
    torch.manual_seed(0)
    batch_size = 512
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               shuffle=True, collate_fn=my_collate)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=1024,
                                               num_workers=2,
                                               shuffle=False, collate_fn=my_collate)
    # model
    model = Model_GRU_xic().to(predifine.device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # epoch 1
    loss_ce = train_one_epoch(train_loader, model, optimizer, loss_fn)

    prob_v, label_v = eval_one_epoch(valid_loader, model)
    prob_v[prob_v >= 0.5] = 1
    prob_v[prob_v < 0.5] = 0
    acc_now = sum(prob_v == label_v) / len(label_v)

    print(f'Epoch[{0}], loss: {loss_ce}, acc: {acc_now}')

    return model


@profile
def utils_model(df_pypro, xics_bank, model):
    prob_v = []
    for _, df_batch in df_pypro.groupby(df_pypro.index // 100000):
        print('\r****** test numbers of XICs: {}/{} finished'.format(df_batch.index[-1] + 1, len(df_pypro)), end='',
              flush=True)

        # xic
        idx = df_batch['xic_idx'].values
        X = xics_bank[idx]

        xics, xics_len = stack_filter_xic(X)
        y = np.zeros(len(xics_len))
        dataset = Model_Dataset_xic(xics, xics_len, y, type='test')

        # dataloader
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=2048,
                                                  num_workers=4,
                                                  shuffle=False, collate_fn=my_collate)

        probs, _ = eval_one_epoch(data_loader, model)

        prob_v.extend(probs)

    print('\r')

    return np.array(prob_v)


@profile
def extract_diann_xics(df_input, mzml):
    pd.options.mode.chained_assignment = None

    xics_v = []

    for _, df in df_input.groupby(df_input.index // 100000):
        print('\rextract xics: {}/{} finished'.format(df.index[-1] + 1, len(df_input)), end='', flush=True)
        # vect
        scans_ms1_rt = mzml.get_ms1_all_rt()
        num_windows = len(mzml.SwathSettings) - 1
        df['query_rt_left'] = df['rt_start'] * 60. - predifine.extend_time
        df['query_rt_right'] = df['rt_stop'] * 60. + predifine.extend_time

        df['idx_start'] = np.abs(df['query_rt_left'].to_numpy().reshape(-1, 1) - scans_ms1_rt).argmin(axis=1) * (
                num_windows + 1)
        df['idx_end'] = np.abs(df['query_rt_right'].to_numpy().reshape(-1, 1) - scans_ms1_rt).argmin(axis=1) * (
                num_windows + 1)
        df['ms2_win_idx'] = np.digitize(df['pr_mz'], mzml.SwathSettings)

        df['xic_num'] = df['query_mz'].apply(len)
        query_mz_v = df['query_mz'].explode().to_numpy().astype(np.float32)

        xic_num_v = df['xic_num'].to_numpy()
        fg_idx_v = [0]
        fg_idx_v.extend(xic_num_v)
        fg_idx_v = np.array(fg_idx_v).cumsum()
        idx_start_v = df['idx_start'].to_numpy()
        idx_end_v = df['idx_end'].to_numpy()
        ms2_win_idx_v = df['ms2_win_idx'].to_numpy()

        for i in range(len(df)):
            fg_idx_start = fg_idx_v[i]
            fg_idx_end = fg_idx_v[i + 1]

            query_mz = query_mz_v[fg_idx_start: fg_idx_end]
            idx_start = idx_start_v[i]
            idx_end = idx_end_v[i]
            ms2_win_idx = ms2_win_idx_v[i]

            xics_ms1, rts_ms1, xics_ms2, rts_ms2 = mzml.get_ms1_ms2_xics_by_lib_mz(idx_start, idx_end,
                                                                                   ms2_win_idx, query_mz,
                                                                                   predifine.MassAccuracyMs1,
                                                                                   predifine.MassAccuracy)
            xics = np.zeros((xic_num_v[i], predifine.target_dim))
            sj_unify_dim(xics_ms1, rts_ms1, xics_ms2, rts_ms2, xics)
            xics = xics[4:, :]  # invalid
            xics_v.append(xics)

    print('\r')
    xics = np.array(xics_v)
    y = (1 - df_input['decoy']).to_numpy()

    return xics, y



def get_diann_pos_neg_xic(df, xics_bank):
    idx = df['xic_idx'].values
    xics = xics_bank[idx]

    y = (1 - df['decoy']).to_numpy()

    return xics, y


def load_diann_params(dir):
    with open(dir) as f:
        content = f.readlines()
    for line in content:
        line = line.strip().split('\t')

        if line[0].find('MassCorrectionMs1') != -1:
            for i in range(1, len(line)):
                predifine.MassCorrectionMs1.append(float(line[i]))
            continue

        if line[0].find('MassCalCenterMs1') != -1:
            for i in range(1, len(line)):
                predifine.MassCalCenterMs1.append(float(line[i]))
            continue

        if line[0].find('MassCalBinsMs1') != -1:
            predifine.MassCalBinsMs1 = int(line[1])
            continue

        if line[0].find('MassAccuracyMs1') != -1:
            predifine.MassAccuracyMs1 = float(line[1]) * 1000000.
            continue

        if line[0].find('MassCorrection') != -1:
            for i in range(1, len(line)):
                predifine.MassCorrection.append(float(line[i]))
            continue

        if line[0].find('MassCalCenter') != -1:
            for i in range(1, len(line)):
                predifine.MassCalCenter.append(float(line[i]))
            continue

        if line[0].find('MassCalBins') != -1:
            predifine.MassCalBins = int(line[1])
            continue

        if line[0].find('MassAccuracy') != -1:
            predifine.MassAccuracy = float(line[1]) * 1000000.
            continue


@jit(nopython=True)
def predicted_mz_ms1(mz, rt, t, MassCalCenterMs1, MassCalBinsMs1):
    s = t[0] * mz * mz
    if (rt <= MassCalCenterMs1[0]):
        s += t[1] + t[2] * mz
    elif (rt >= MassCalCenterMs1[MassCalBinsMs1 - 1]):
        s += t[1 + (MassCalBinsMs1 - 1) * 2] + t[2 + (MassCalBinsMs1 - 1) * 2] * mz
    else:
        for i in range(1, MassCalBinsMs1):
            if (rt < MassCalCenterMs1[i]):
                u = rt - MassCalCenterMs1[i - 1]
                v = MassCalCenterMs1[i] - rt
                w = u + v
                if w > 0.000000001:
                    s += ((t[1 + (i - 1) * 2] + t[2 + (i - 1) * 2] * mz) * v + (
                            t[1 + i * 2] + t[2 + i * 2] * mz) * u) / w
                break
    return s + mz


@jit(nopython=True)
def predicted_mz(mz, rt, t, MassCalCenter, MassCalBins):
    s = t[0] * mz * mz
    if (rt <= MassCalCenter[0]):
        s += t[1] + t[2] * mz
    elif (rt >= MassCalCenter[MassCalBins - 1]):
        s += t[1 + (MassCalBins - 1) * 2] + t[2 + (MassCalBins - 1) * 2] * mz
    else:
        for i in range(1, MassCalBins):
            if (rt < MassCalCenter[i]):
                u = rt - MassCalCenter[i - 1]
                v = MassCalCenter[i] - rt
                w = u + v
                if w > 0.000000001:
                    s += ((t[1 + (i - 1) * 2] + t[2 + (i - 1) * 2] * mz) * v + (
                            t[1 + i * 2] + t[2 + i * 2] * mz) * u) / w
                break
    return s + mz

def gen_mutate_dict():
    mutate_mass_change = {}
    aa_to_mass = {'A': 71.037114, 'C': 103.009185, 'D': 115.026943, 'E': 129.042593,
                  'F': 147.068414, 'G': 57.021464, 'H': 137.058912, 'I': 113.084064,
                  'K': 128.094963, 'L': 113.084064, 'M': 131.040485, 'N': 114.042927,
                  'P': 97.052764, 'Q': 128.058578, 'R': 156.101111, 'S': 87.032028,
                  'T': 101.047679, 'V': 99.068414, 'W': 186.079313, 'Y': 163.06332}

    for old, new in zip('GAVLIFMPWSCTYHKRQEND', 'LLLVVLLLLTSSSSLLNDQE'):
        mass_old = aa_to_mass[old]
        mass_new = aa_to_mass[new]
        mass_delta = mass_new - mass_old
        mutate_mass_change[old] = mass_delta

    return mutate_mass_change


def mutate_one_aa(mutate_mass_change, v):
    f = operator.itemgetter(*v)
    mass_change = f(mutate_mass_change)
    return mass_change


def preprocess_info_df(df):
    df['pr_charge'] = df['pr_id'].str[-1].astype(int)

    # mutate
    mutate_mass_change = gen_mutate_dict()
    df['simple_seq'] = df['pr_id'].str[0:-1]
    df['simple_seq'] = df['simple_seq'].replace(['C\(UniMod:4\)', 'M\(UniMod:35\)'], ['C', 'm'], regex=True)
    assert df['simple_seq'].str.contains('UniMod').sum() == 0
    df['seq_len'] = df['simple_seq'].str.len()
    df['second_bone_C'] = df['simple_seq'].str[-2].str.upper()
    df['second_bone_N'] = df['simple_seq'].str[1].str.upper()
    df['C_shift'] = mutate_one_aa(mutate_mass_change, df['second_bone_C'])
    df['N_shift'] = mutate_one_aa(mutate_mass_change, df['second_bone_N'])

    mass_idx_v = [0]
    mass_idx_v.extend(df['seq_len'])

    # pr_mz const
    mass_neutron = 1.0033548378
    df['pr_mz_1'] = (df['pr_mz'] * df['pr_charge'] + mass_neutron) / df['pr_charge']
    df['pr_mz_2'] = (df['pr_mz'] * df['pr_charge'] + 2 * mass_neutron) / df['pr_charge']

    # to_numpy
    pr_mz_v = df['pr_mz'].to_numpy()
    pr_mz_1_v = df['pr_mz_1'].to_numpy()
    pr_mz_2_v = df['pr_mz_2'].to_numpy()
    fg_mz_v = df['fg_mz'].to_numpy()
    rt_v = df['rt'].to_numpy()

    pr_mz_pred_v, query_mz_v = [], []

    for i in range(len(df)):

        pr_mz = pr_mz_v[i]
        pr_mz_1 = pr_mz_1_v[i]
        pr_mz_2 = pr_mz_2_v[i]
        query_pr_mz = np.array([pr_mz, pr_mz_1, pr_mz_2])

        # fg_mz from lib
        fg_mz = np.fromstring(fg_mz_v[i], sep=';')

        query_fg_mz = np.concatenate([[pr_mz], fg_mz])

        rt = rt_v[i]  # unit: min
        assert rt < 60. * 4

        query_pr_mz = [predicted_mz_ms1(mz, rt,
                                        np.array(predifine.MassCorrectionMs1),
                                        np.array(predifine.MassCalCenterMs1),
                                        predifine.MassCalBinsMs1) for mz in query_pr_mz]
        query_fg_mz = [predicted_mz(mz, rt,
                                    np.array(predifine.MassCorrection),
                                    np.array(predifine.MassCalCenter),
                                    predifine.MassCalBins) for mz in query_fg_mz]
        pr_mz_pred = query_pr_mz[0]
        query_mz = np.concatenate([query_pr_mz, query_fg_mz])

        pr_mz_pred_v.append(pr_mz_pred)
        query_mz_v.append(query_mz)

    df['pr_mz_pred'] = pr_mz_pred_v
    df['query_mz'] = query_mz_v

    return df


@jit(nopython=True)
def sj_unify_dim(xics_ms1, rts_ms1, xics_ms2, rts_ms2, xics):
    target_dim = xics.shape[1]
    rt_start = rts_ms1[0]
    rt_end = rts_ms1[-1]
    delta_rt = (rt_end - rt_start) / (target_dim - 1)

    idx_ms1 = 1
    idx_ms2 = 1
    for i in range(target_dim):
        x = rt_start + i * delta_rt
        # ms1
        if x > rts_ms1[idx_ms1]:
            idx_ms1 += 1
        x0 = rts_ms1[idx_ms1 - 1]
        x1 = rts_ms1[idx_ms1]
        for j in range(xics_ms1.shape[0]):
            y0 = xics_ms1[j, idx_ms1 - 1]
            y1 = xics_ms1[j, idx_ms1]
            y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
            xics[j, i] = y
        # ms2
        if x > rts_ms2[idx_ms2]:
            idx_ms2 += 1
        x0 = rts_ms2[idx_ms2 - 1]
        x1 = rts_ms2[idx_ms2]
        for j in range(xics_ms2.shape[0]):
            y0 = xics_ms2[j, idx_ms2 - 1]
            y1 = xics_ms2[j, idx_ms2]
            y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
            xics[j + 3, i] = y


def add_alpha_xic_score(df, mzml):
    df = preprocess_info_df(df)
    # xic_idx
    df['xic_idx'] = np.arange(len(df))

    # extract all
    t1 = time.time()
    xics_bank, _ = extract_diann_xics(df, mzml)

    print(f'training Alpha-XIC')
    t1 = time.time()
    xics, y = get_diann_pos_neg_xic(df, xics_bank)
    model = train_model(xics, y, random_state=12345)
    probs = utils_model(df, xics_bank, model)
    print('Alpha-XIC training finished. time: {:.2f}s'.format(time.time() - t1))

    df['score_alpha_xic'] = probs

    return df