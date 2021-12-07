#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   main.py
@Author :   Song
@Time   :   2021/6/30 9:47
@Contact:   songjian@westlake.edu.cn
@intro  : 
'''
import time
import pandas as pd
import utils_tri
import utils_xic
import ms
import warnings
import argparse
from pathlib import Path
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


try:
    profile
except NameError:
    profile = lambda x: x


def get_arg():
    parser = argparse.ArgumentParser('')
    parser.add_argument('-ws', '--ws', required=True, help='specify the absolute path of your workspace')
    parser.add_argument('-xic', '--xic', action='store_true', help='specify whether adding alpha_xic score')
    parser.add_argument('-tri', '--tri', action='store_true', help='specify whether adding alpha_tri score')
    args = parser.parse_args()

    if args.xic is False and args.tri is False:
        raise Exception('-xic or -tri have to one at least')
    return Path(args.ws), args.xic, args.tri


@profile
def main():
    ws, is_xic, is_tri = get_arg()
    path_set = utils_tri.choose_ws(ws)
    for p in path_set:
        print(p)
    path_params, ms_path, path_diann, path_info, path_info_q, path_lib = path_set

    # load data shared by alpha-xic and alpha-tri
    utils_xic.load_diann_params(path_params)
    mzml = ms.load_ms(ms_path, type='DIA')
    df = pd.read_csv(path_info, sep='\t')
    df_prosit = pd.read_pickle(path_lib)
    raw_cols = list(df.columns)

    # append score
    if is_tri and not is_xic:
        cols = raw_cols + ['score_alpha_tri']
        df = utils_tri.add_alpha_tri_score(df, mzml, df_prosit)
    if is_xic and not is_tri:
        cols = raw_cols + ['score_alpha_xic']
        df = utils_xic.add_alpha_xic_score(df, mzml)
    if is_xic and is_tri:
        cols = raw_cols + ['score_alpha_tri', 'score_alpha_xic']
        df = utils_xic.add_alpha_xic_score(df, mzml)
        df = utils_tri.add_alpha_tri_score(df, mzml, df_prosit)

    # q value
    df = utils_tri.get_prophet_result(df)

    # save
    cols = cols + ['cscore', 'q_value', 'Precursor.Quantity']
    df_diann = pd.read_csv(path_diann, sep='\t', usecols=['Precursor.Id', 'Precursor.Quantity'])
    df_result = pd.merge(df, df_diann, left_on='pr_id', right_on='Precursor.Id')
    df_result[cols].to_csv(path_info_q, sep='\t', index=False)


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f'finished. {(time.time() - t0) / 60.:.3f}min')
