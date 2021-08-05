#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   main.py
@Author :   Song
@Time   :   2021/6/30 9:47
@Contact:   songjian@westlake.edu.cn
@intro  : 
'''
import sys
import time
import pandas as pd
import utils
import ms
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

import torch
import torch.nn
from torch.utils.data.dataset import TensorDataset

from Model_GRU import Model_GRU
from Alpha_Dataloader import Alpha_Dataset
import predifine
import torch.nn.utils.rnn as rnn_utils

try:
    profile
except NameError:
    profile = lambda x: x


def my_collate(items):
    batch_tri, batch_charge, batch_seq_len, batch_y = zip(*items)

    batch_tri_len = list(map(len, batch_tri))
    batch_tri_len = torch.tensor(batch_tri_len)
    batch_tri = rnn_utils.pad_sequence(batch_tri, batch_first=True, padding_value=-1)
    batch_charge = torch.tensor(batch_charge)
    batch_seq_len = torch.tensor(batch_seq_len)
    batch_y = torch.tensor(batch_y)

    return batch_tri, batch_tri_len, batch_charge, batch_seq_len, batch_y


@profile
def main():
    path_set = utils.choose_ws(sys.argv[1])
    for p in path_set:
        print(p)
    path_params, ms_path, path_diann, path_info, path_info_q, path_lib = path_set
    utils.load_diann_params(path_params)  # time unit: min

    df_prosit = pd.read_pickle(path_lib)
    df = pd.read_csv(path_info, sep='\t')  # time unit: min
    assert df['pr_id'].isin(df_prosit['pr_id']).all()

    raw_cols = list(df.columns)

    df = df.merge(df_prosit, on='pr_id', how='left')
    df = utils.preprocess_info_df(df)

    # load mz and prepare xic params
    mzml = ms.load_ms(ms_path, type='DIA')
    df = utils.prepare_xic_params(df, mzml)

    # construct feature[spectrum_pred, spectrum_m, pcc]
    X = utils.construct_features(df, mzml)
    X_pr_charge = df['pr_charge'].to_numpy()
    X_seq_len = df['seq_len'].to_numpy()
    y = 1 - df['decoy'].to_numpy()

    # learn from feature
    model = Model_GRU().to(predifine.device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_dataset = Alpha_Dataset(X, X_pr_charge, X_seq_len, y)
    eval_dataset = Alpha_Dataset(X, X_pr_charge, X_seq_len, y)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                               num_workers=2, shuffle=True, collate_fn=my_collate)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1024,
                                              num_workers=2, shuffle=False, collate_fn=my_collate)

    utils.train_one_epoch(train_loader, model, optimizer, loss_fn)
    print('trian finieshed.')

    # append score
    df['score_alpha_prosit'] = utils.eval_one_epoch(eval_loader, model)
    median_target = df[df['decoy'] == 0]['score_alpha_prosit'].median()
    median_decoy = df[df['decoy'] == 1]['score_alpha_prosit'].median()
    print(f'median_target: {median_target}, median_decoy: {median_decoy}')

    # q value
    df = utils.get_prophet_result(df)

    # save
    cols = raw_cols + ['score_alpha_prosit', 'cscore', 'q_value', 'Precursor.Quantity']
    df_diann = pd.read_csv(path_diann, sep='\t', usecols=['Precursor.Id', 'Precursor.Quantity'])
    df_result = pd.merge(df, df_diann, left_on='pr_id', right_on='Precursor.Id')
    df_result[cols].to_csv(path_info_q, sep='\t', index=False)


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f'finished. {(time.time() - t0) / 60.:3f}min')
