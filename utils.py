#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   utils.py
@Author :   Song
@Time   :   2021/6/30 9:48
@Contact:   songjian@westlake.edu.cn
@intro  : 
'''
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.cuda
import torch.backends.cudnn
import multiprocessing as mp

from pathlib import Path
import predifine
from numba import jit
import operator
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

try:
    profile
except:
    profile = lambda x: x


def choose_ws(ws):
    ws = Path(ws)
    ms_path = list(ws.glob('*.mzML'))[0]
    path_params = ws / 'cal_params.tsv'
    path_diann = ws / 'diann_out.tsv'
    path_info = ws / 'diann_info.tsv'
    path_info_q = ws / 'alpha_out.tsv'
    lib_path = list(ws.glob('*.pkl'))[0]
    return path_params, ms_path, path_diann, path_info, path_info_q, lib_path


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

    predifine.MassCorrection = np.array(predifine.MassCorrection)
    predifine.MassCalCenter = np.array(predifine.MassCalCenter)
    predifine.MassCorrectionMs1 = np.array(predifine.MassCorrectionMs1)
    predifine.MassCalCenterMs1 = np.array(predifine.MassCalCenterMs1)

    if predifine.MassAccuracy < 20.:
        predifine.MassAccuracy = 20.
    print(f'MassAccuracy MS1: {predifine.MassAccuracyMs1}ppm, MS2: {predifine.MassAccuracy}ppm')


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


def pad_seq_to_mass(simple_seq):
    s = simple_seq.str.cat()
    s = list(s)

    f = operator.itemgetter(*s)
    paded_mass = f(predifine.g_aa_to_mass)

    return paded_mass


@jit(nopython=True)
def calculate_possible_mz(masses, pr_charge, decoy, C_shift, N_shift):
    mass_proton = 1.007276466771
    mass_h2o = 18.0105650638

    pr_len = len(masses)
    fg_charge_max = min(pr_charge, 2)

    masses_forward = masses.cumsum()
    masses_backward = masses[::-1].cumsum()

    fg_num = (pr_len - 1) * fg_charge_max * 2 - 4 * fg_charge_max
    fg_mz = np.zeros(fg_num)

    idx = 0
    # y3_1, y3_2, b3_1, b3_2
    for fg_len in range(3, pr_len):
        for y_or_b in [0, 1]:
            for fg_charge in range(1, fg_charge_max + 1):
                if y_or_b == 1:  # b, forward, minus a h2o, C_shift
                    product_mass = masses_forward[fg_len - 1] - (fg_len - 1) * mass_h2o - mass_h2o + decoy * N_shift
                else:  # y
                    product_mass = masses_backward[fg_len - 1] - (fg_len - 1) * mass_h2o + decoy * C_shift
                product_mz = (product_mass + fg_charge * mass_proton) / fg_charge
                fg_mz[idx] = product_mz
                idx = idx + 1
    return fg_mz


@profile
def preprocess_info_df(df):
    df['pr_charge'] = df['pr_id'].str[-1].astype(int)

    mutate_mass_change = gen_mutate_dict()
    df['simple_seq'] = df['pr_id'].str[0:-1]
    df['simple_seq'] = df['simple_seq'].replace(['C\(UniMod:4\)', 'M\(UniMod:35\)'], ['C', 'm'], regex=True)
    assert df['simple_seq'].str.contains('UniMod').sum() == 0
    df['seq_len'] = df['simple_seq'].str.len()
    df['pr_mz'] = df['pr_mz'].astype(np.float32)
    df['second_bone_C'] = df['simple_seq'].str[-2].str.upper()
    df['second_bone_N'] = df['simple_seq'].str[1].str.upper()
    df['C_shift'] = mutate_one_aa(mutate_mass_change, df['second_bone_C'])
    df['N_shift'] = mutate_one_aa(mutate_mass_change, df['second_bone_N'])

    aa_mass_v = np.array(pad_seq_to_mass(df.simple_seq))
    mass_idx_v = [0]
    mass_idx_v.extend(df['seq_len'])
    mass_idx_v = np.array(mass_idx_v).cumsum()

    # to_numpy
    decoy_v = df['decoy'].to_numpy()
    C_shift_v = df['C_shift'].to_numpy()
    N_shift_v = df['N_shift'].to_numpy()
    pr_charge_v = df['pr_charge'].to_numpy()
    rt_v = df['rt'].to_numpy()
    assert (rt_v < 60. * 5).all()  # min
    fg_mz_v = []

    for i in range(len(df)):
        decoy = decoy_v[i]
        C_shift = C_shift_v[i]
        N_shift = N_shift_v[i]

        # fg_mz from enumerate
        mass_idx_start = mass_idx_v[i]
        mass_idx_end = mass_idx_v[i + 1]
        aa_mass = aa_mass_v[mass_idx_start:mass_idx_end]
        pr_charge = pr_charge_v[i]
        fg_mz = calculate_possible_mz(aa_mass, pr_charge, decoy, C_shift, N_shift)

        rt = rt_v[i]  # unit: min
        fg_mz = [predicted_mz(mz, rt,
                              predifine.MassCorrection,
                              predifine.MassCalCenter,
                              predifine.MassCalBins) for mz in fg_mz]

        fg_mz_v.append(fg_mz)

    df['fg_mz'] = fg_mz_v
    return df


@profile
def prepare_xic_params(df_input, mzml):
    pd.options.mode.chained_assignment = None
    df_result = pd.DataFrame()

    for _, df in df_input.groupby(df_input.index // 50000):
        scans_ms1_rt = mzml.get_ms1_all_rt()
        num_windows = len(mzml.SwathSettings) - 1
        df['query_rt_left'] = df['rt_start'] * 60.
        df['query_rt_right'] = df['rt_stop'] * 60.

        df['idx_start'] = np.abs(df['query_rt_left'].to_numpy().reshape(-1, 1) - scans_ms1_rt).argmin(axis=1) * (
                num_windows + 1)
        df['idx_end'] = np.abs(df['query_rt_right'].to_numpy().reshape(-1, 1) - scans_ms1_rt).argmin(axis=1) * (
                num_windows + 1)
        df['ms2_win_idx'] = np.digitize(df['pr_mz'], mzml.SwathSettings)

        df_result = pd.concat([df_result, df], axis=0, ignore_index=True)

    return df_result


@jit(nopython=True)
def find_ok_matches(scan_mz, scan_intensity, mz_query, ppm):
    num_basis = scan_mz.shape[0]
    num_samples = mz_query.shape[0]
    result = np.zeros(len(mz_query), dtype=np.float32)

    for i in range(num_samples):
        sp_i = mz_query[i]
        low = 0
        high = num_basis - 1
        best_j = 0
        if scan_mz[low] == sp_i:
            best_j = low
        elif scan_mz[high] == sp_i:
            best_j = high
        else:
            while high - low > 1:
                mid = int((low + high) / 2)
                if scan_mz[mid] == sp_i:
                    best_j = mid
                    break
                if scan_mz[mid] < sp_i:
                    low = mid
                else:
                    high = mid
            if best_j == 0:
                if abs(scan_mz[low] - sp_i) < abs(scan_mz[high] - sp_i):
                    best_j = low
                else:
                    best_j = high
        # find first match in list !
        while best_j > 0:
            if scan_mz[best_j - 1] == scan_mz[best_j]:
                best_j = best_j - 1
            else:
                break

        mz_nearest = scan_mz[best_j]
        if abs(sp_i - mz_nearest) * 1000000. < ppm * sp_i:
            result[i] = scan_intensity[best_j]

    return result


@jit(nopython=True)
def smooth_xic(xics):
    result = np.zeros_like(xics)
    result[:, 0] = 2. / 3. * xics[:, 0] + 1. / 3. * xics[:, 1]
    result[:, -1] = 2. / 3. * xics[:, -1] + 1. / 3. * xics[:, -2]
    for i in range(1, xics.shape[1] - 1):
        result[:, i] = 0.5 * xics[:, i] + 0.25 * (xics[:, i - 1] + xics[:, i + 1])

    return result


@jit(nopython=True)
def make_perfect_shape(rt_left, rt_apex, rt_right, dim):
    output = np.zeros(dim)
    epsilon = 0.01
    points_left = round(dim * (rt_apex - rt_left) / (rt_right - rt_left))
    if points_left < 2:
        points_left = 2
    if dim - points_left < 2:
        points_left = dim - 2

    # left
    u = points_left - 1
    sig2 = -u ** 2 / math.log(epsilon) / 2.
    for i in range(points_left):
        output[i] = math.exp(-(i - u) ** 2 / 2. / sig2)

    # right
    sig2 = -(dim - 1 - u) ** 2 / math.log(epsilon) / 2.
    for i in range(points_left, dim):
        output[i] = math.exp(-(i - u) ** 2 / 2. / sig2)

    return output


@jit(nopython=True)
def calculate_sa_v(xics, theory_shape):
    sa_v = -np.ones(xics.shape[0])
    y_max = theory_shape.max()
    for i in range(xics.shape[0]):
        x = xics[i]
        x_max = x.max()
        sa = calculate_sa(x, x_max, theory_shape, y_max)
        sa_v[i] = sa
    return sa_v


@profile
def construct_features(df, mzml):
    df['fg_num'] = df['fg_mz'].apply(len)
    fg_mz_v = df['fg_mz'].explode().to_numpy().astype(np.float32)

    fg_num_v = df['fg_num'].to_numpy()
    fg_idx_v = [0]
    fg_idx_v.extend(fg_num_v)
    fg_idx_v = np.array(fg_idx_v).cumsum()
    idx_start_v = df['idx_start'].to_numpy()
    idx_end_v = df['idx_end'].to_numpy()
    ms2_win_idx_v = df['ms2_win_idx'].to_numpy()
    rt_left_v = df['rt_start'].to_numpy() * 60.
    rt_right_v = df['rt_stop'].to_numpy() * 60.
    rt_apex_v = df['rt'].to_numpy() * 60.
    spectrum_pred_v = df['spectrum_pred'].to_numpy()

    feature_v = []
    for i in range(len(df)):
        if i % 50000 == 0:
            print(i)
        fg_idx_start = fg_idx_v[i]
        fg_idx_end = fg_idx_v[i + 1]
        rt_left = rt_left_v[i]
        rt_apex = rt_apex_v[i]
        rt_right = rt_right_v[i]
        fg_mz = fg_mz_v[fg_idx_start: fg_idx_end]
        idx_start = idx_start_v[i]
        idx_end = idx_end_v[i]
        ms2_win_idx = ms2_win_idx_v[i]

        xics, rts = mzml.get_ms2_xics_by_fg_mz(idx_start, idx_end,
                                               ms2_win_idx, fg_mz,
                                               predifine.MassAccuracy)
        xics = smooth_xic(xics)
        perfect_shape = make_perfect_shape(rt_left, rt_apex, rt_right, xics.shape[1])
        xics_sa = calculate_sa_v(xics, perfect_shape)

        # feature
        apex_idx = int(xics.shape[1] * (rt_apex - rt_left) / (rt_right - rt_left))
        xics_apex = xics[:, apex_idx]

        spectrum_m = xics_apex / (xics_apex.max() + 0.0000001)

        spectrum_pred = spectrum_pred_v[i]
        spectrum_pred = spectrum_pred[spectrum_pred > -1]

        # norm
        spectrum_pred = spectrum_pred / (spectrum_pred.max() + 0.0000001)
        spectrum_m = spectrum_m / (spectrum_m.max() + 0.0000001)

        feature = np.concatenate([spectrum_pred, spectrum_m, xics_sa])

        feature_v.append(feature)

    feature_v = np.array(feature_v)

    return feature_v


@profile
def construct_score_sa_weight(df, mzml):
    df['fg_num'] = df['fg_mz'].apply(len)
    fg_mz_v = df['fg_mz'].explode().to_numpy().astype(np.float32)

    fg_num_v = df['fg_num'].to_numpy()
    fg_idx_v = [0]
    fg_idx_v.extend(fg_num_v)
    fg_idx_v = np.array(fg_idx_v).cumsum()
    idx_start_v = df['idx_start'].to_numpy()
    idx_end_v = df['idx_end'].to_numpy()
    ms2_win_idx_v = df['ms2_win_idx'].to_numpy()
    rt_left_v = df['rt_start'].to_numpy() * 60.
    rt_right_v = df['rt_stop'].to_numpy() * 60.
    rt_apex_v = df['rt'].to_numpy() * 60.
    spectrum_pred_v = df['spectrum_pred'].to_numpy()
    feature_v = []
    for i in range(len(df)):
        if i % 50000 == 0:
            print(i)
        fg_idx_start = fg_idx_v[i]
        fg_idx_end = fg_idx_v[i + 1]
        rt_left = rt_left_v[i]
        rt_apex = rt_apex_v[i]
        rt_right = rt_right_v[i]
        fg_mz = fg_mz_v[fg_idx_start: fg_idx_end]
        idx_start = idx_start_v[i]
        idx_end = idx_end_v[i]
        ms2_win_idx = ms2_win_idx_v[i]

        xics, rts = mzml.get_ms2_xics_by_fg_mz(idx_start, idx_end,
                                               ms2_win_idx, fg_mz,
                                               predifine.MassAccuracy)
        xics = smooth_xic(xics)
        perfect_shape = make_perfect_shape(rt_left, rt_apex, rt_right, xics.shape[1])
        xics_sa = calculate_sa_v(xics, perfect_shape)

        # 构建feature，需要按照子离子的排序扩充到116维度
        apex_idx = int(xics.shape[1] * (rt_apex - rt_left) / (rt_right - rt_left))
        xics_apex = xics[:, apex_idx]
        xics_apex = xics_apex / (xics_apex.max() + 0.0000001)

        spectrum_pred = spectrum_pred_v[i]
        spectrum_pred = spectrum_pred[spectrum_pred > -1]

        sa = calculate_sa_weight(spectrum_pred, xics_apex, xics_sa)
        feature_v.append(sa)

    feature_v = np.array(feature_v)

    return feature_v


@jit(nopython=True)
def calculate_sa(x, x_max, y, y_max):
    e = 0.000001
    s = 0.
    norm_x = 0.
    norm_y = 0.
    for i in range(len(x)):
        xx = x[i] / (x_max + e)
        yy = y[i] / (y_max + e)
        norm_x += xx ** 2
        norm_y += yy ** 2
        s += xx * yy
    norm_x = math.sqrt(norm_x) + e
    norm_y = math.sqrt(norm_y) + e
    sa = 1 - 2 * math.acos(s / norm_x / norm_y) / np.pi
    return sa


@jit(nopython=True)
def calculate_sa_weight(x, y, weight):
    y = y * weight
    x_max = x.max()
    y_max = y.max()
    e = 0.000001
    s = 0.
    norm_x = 0.
    norm_y = 0.
    for i in range(len(x)):
        xx = x[i] / (x_max + e)
        yy = y[i] / (y_max + e)
        norm_x += xx ** 2
        norm_y += yy ** 2
        s += xx * yy
    norm_x = math.sqrt(norm_x) + e
    norm_y = math.sqrt(norm_y) + e
    sa = 1 - 2 * math.acos(s / norm_x / norm_y) / np.pi
    return sa


def process_worker(i, X, y):
    mlps = [MLPClassifier(max_iter=1, shuffle=True, random_state=i * 3 + ii,
                          learning_rate_init=0.003, solver='adam', batch_size=50,
                          activation='tanh', hidden_layer_sizes=(25, 20, 15, 10, 5)) for ii in range(3)]
    for mlp in mlps:
        mlp.fit(X, y)
    nn_score = [mlp.predict_proba(X)[:, 1] for mlp in mlps]
    return nn_score


def get_prophet_result(df):
    t = time.time()
    col_idx = df.columns.str.startswith('score_')
    X = df.loc[:, col_idx].to_numpy()
    y = 1 - df['decoy'].to_numpy()
    print(
        f'Training the neural network of DIA-NN: {(y == 1).sum()} targets, {(y == 0).sum()} decoys, scores num: {sum(col_idx)}')

    # norm
    X = preprocessing.scale(X)

    # multiple process
    pool = mp.Pool(4)
    results = [pool.apply_async(process_worker, args=(i, X, y)) for i in range(4)]
    results = [r.get() for r in results]  # get同步进程。result是四个字典
    pool.close()
    pool.join()
    nn_score = np.vstack(results).mean(axis=0)

    # q value
    df['cscore'] = nn_score
    df = df.sort_values(by='cscore', ascending=False, ignore_index=True)

    target_num = (df.decoy == 0).cumsum()
    decoy_num = (df.decoy == 1).cumsum()

    target_num[target_num == 0] = 1
    decoy_num[decoy_num == 0] = 1
    df['q_value'] = decoy_num / target_num
    df['q_value'] = df['q_value'][::-1].cummin()

    # log
    id_001 = ((df['q_value'] <= 0.001) & (df['decoy'] == 0)).sum()
    id_010 = ((df['q_value'] <= 0.01) & (df['decoy'] == 0)).sum()
    id_100 = ((df['q_value'] <= 0.1) & (df['decoy'] == 0)).sum()
    id_500 = ((df['q_value'] <= 0.5) & (df['decoy'] == 0)).sum()
    print(f'Number of IDs at 50%, 10%, 1%, 0.1% FDR: {id_500}, {id_100}, {id_010}, {id_001}')
    print(f'DIA-NN neural network time: {time.time() - t:.3f}s')

    return df


def train_one_epoch(train_loader, model, optimizer, loss_fn):
    model.train()

    device = predifine.device
    batch_loss_v = 0.
    for batch_idx, (batch_tri, batch_tri_len, batch_pr_charge, batch_seq_len, batch_y) in enumerate(train_loader):
        batch_tri = batch_tri.float().to(device)
        batch_tri_len = batch_tri_len.long().to(device)
        batch_pr_charge = batch_pr_charge.long().to(device)
        batch_seq_len = batch_seq_len.long().to(device)
        batch_y = batch_y.long().to(device)
        optimizer.zero_grad()

        # forward
        prob = model(batch_tri, batch_tri_len, batch_pr_charge, batch_seq_len)
        # loss
        batch_loss = loss_fn(prob, batch_y)
        # back
        batch_loss.backward()
        # update
        optimizer.step()
        batch_loss_v += batch_loss.item()

    epoch_loss = (batch_loss_v) / (batch_idx + 1)
    return epoch_loss


def eval_one_epoch(evalloader, model):
    model.eval()
    device = predifine.device

    prob_v = []
    for batch_idx, (batch_tri, batch_tri_len, batch_pr_charge, batch_seq_len, batch_y) in enumerate(evalloader):
        batch_tri = batch_tri.float().to(device)
        batch_tri_len = batch_tri_len.long().to(device)
        batch_pr_charge = batch_pr_charge.long().to(device)
        batch_seq_len = batch_seq_len.long().to(device)

        # forward
        prob = model(batch_tri, batch_tri_len, batch_pr_charge, batch_seq_len)
        prob = torch.softmax(prob.view(-1, 2), 1)
        probs = prob[:, 1].tolist()
        prob_v.extend(probs)

    return prob_v
