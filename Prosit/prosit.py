#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   main_prosit.py
@Author :   Song
@Time   :   2021/12/7 13:05
@Contact:   songjian@westlake.edu.cn
@intro  :   Predicting the MS2 for peptides from spectral library
'''
import argparse
from pathlib import Path
import pandas as pd
import tensorize, prediction, model

prosit_idx_to_anno = {}
for idx in range(174):
    anno_y_b = 'y' if idx % 6 <= 2 else 'b'
    anno_site = (idx // 6) + 1
    anno_charge = (idx % 3) + 1
    anno = anno_y_b + str(anno_site) + '_' + str(anno_charge)
    prosit_idx_to_anno[idx] = anno

def polish_lib(df_lib):
    # no decoy
    assert (df_lib['decoy'] == 0).all(), 'This library should not contains decoy peptides.'
    # no PTM except for M(ox)
    df_lib['simple_seq'] = df_lib['FullUniModPeptideName'].replace(['C\(UniMod:4\)', 'M\(UniMod:35\)'], ['C', 'm'],
                                                                   regex=True)
    df_lib = df_lib[~df_lib['simple_seq'].str.contains('UniMod')]
    assert (df_lib['simple_seq'].str.contains('UniMod') == 0).all()
    # only [BJOUXZ]
    df_lib = df_lib[~df_lib['simple_seq'].str.contains('[BJOUXZ]', regex=True)]
    # charge [1, 4]
    df_lib = df_lib[df_lib['PrecursorCharge'] < 5]
    # length [7, 30]
    df_lib = df_lib[(df_lib['simple_seq'].str.len() > 6) & (df_lib['simple_seq'].str.len() < 31)]
    # drop duplicates
    df_lib['pr_id'] = df_lib['FullUniModPeptideName'] + df_lib['PrecursorCharge'].astype(str)
    df_lib = df_lib.drop_duplicates(subset=['pr_id'])
    df_lib = df_lib.reset_index(drop=True)

    return df_lib

def get_arg():
    parser = argparse.ArgumentParser('')
    parser.add_argument('-lib', '--lib', required=True, help='specify the absolute path of your spectral library')
    parser.add_argument('-nce', '--nce', type=int, default=35, help='specify the NCE for Prosit model')
    args = parser.parse_args()

    return Path(args.lib), args.nce

if __name__ == '__main__':
    path_lib, NCE = get_arg()

    df_lib = pd.read_csv(path_lib, sep='\t')
    dir_out = path_lib.parent/(path_lib.stem + '.pkl')

    df_lib = polish_lib(df_lib)

    df_prosit = pd.DataFrame()
    df_prosit['modified_sequence'] = df_lib['simple_seq'].replace('m', 'M(ox)', regex=True)
    df_prosit['collision_energy'] = NCE
    df_prosit['precursor_charge'] = df_lib['PrecursorCharge']
    df_prosit = df_prosit.reset_index(drop=True)

    model, model_config = model.load(r'model', trained=True)

    tensor = tensorize.tensor = tensorize.peptidelist(df_prosit)
    prosit_result = prediction.predict(tensor, model, model_config)

    m_pred_intensity = prosit_result['intensities_pred']
    cols = [i for i in range(len(prosit_idx_to_anno)) if i % 3 != 2]
    m_pred_intensity = m_pred_intensity[:, cols] # remove charge 3 fragments
    m_pred_intensity = m_pred_intensity[:, 8:] # remove short fragments

    df_result = pd.DataFrame()
    df_result['pr_id'] = df_lib['FullUniModPeptideName'] + df_lib['PrecursorCharge'].astype(str)
    df_result['spectrum_pred'] = list(m_pred_intensity)

    df_result.to_pickle(dir_out)