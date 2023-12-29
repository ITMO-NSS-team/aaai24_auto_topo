import os
from pathlib import Path

import numpy as np
import pandas as pd
from sktime.dists_kernels import DtwDist
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error


def MASE(A, F, y_train):
    return mean_absolute_scaled_error(A, F, y_train=y_train)


ids_to_proceed = []

targets = {}

global_mases = {}

autogluon = 'autogluon'
evo_ws_selection = 'evo_ws_selection'
nbeats = 'NBEATS'
only_evo = 'only_evo'
static_pipeline_topo_ws_selection = 'static_pipeline_topo_ws_selection'
topo_ws_selection_evo = 'topo_ws_selection_evo'

concated_df = pd.DataFrame()
for i in os.listdir(Path('data')):
    concated_df = pd.concat([concated_df, pd.read_csv(Path('data', i))], axis=0)

mases_f = {}
for i in os.listdir(only_evo):
    if i.endswith('csv'):
        idx = i.split('_')[0]
        ids_to_proceed.append(idx)
        df = pd.read_csv(Path(only_evo, i))
        real_values = concated_df[concated_df['label'] == idx]['value'].values
        targets[idx] = df['value']
        mases_f[idx] = MASE(df['value'], df['predict'], real_values)
global_mases[only_evo] = mases_f

mases_f = {}
for i in os.listdir(topo_ws_selection_evo):
    if i.endswith('csv'):
        idx = i.split('_')[0]
        ids_to_proceed.append(idx)
        df = pd.read_csv(Path(topo_ws_selection_evo, i))
        real_values = concated_df[concated_df['label'] == idx]['value'].values
        targets[idx] = df['value']
        mases_f[idx] = MASE(df['value'], df['predict'], real_values)
global_mases[topo_ws_selection_evo] = mases_f

mases_f = {}
for i in os.listdir(evo_ws_selection):
    if i.endswith('csv'):
        idx = i.split('_')[0]
        ids_to_proceed.append(idx)
        df = pd.read_csv(Path(evo_ws_selection, i))
        real_values = concated_df[concated_df['label'] == idx]['value'].values
        targets[idx] = df['value']
        mases_f[idx] = MASE(df['value'], df['predict'], real_values)
global_mases[evo_ws_selection] = mases_f

mases_f = {}
for i in os.listdir(static_pipeline_topo_ws_selection):
    if i.endswith('csv'):
        idx = i.split('_')[0]
        ids_to_proceed.append(idx)
        df = pd.read_csv(Path(static_pipeline_topo_ws_selection, i))
        real_values = concated_df[concated_df['label'] == idx]['value'].values
        targets[idx] = df['value']
        mases_f[idx] = MASE(df['value'], df['predict'], real_values)
global_mases[static_pipeline_topo_ws_selection] = mases_f

mases_f = {}
for i in os.listdir(nbeats):
    if i.endswith('csv'):
        idx = i.split('_')[0]
        ids_to_proceed.append(idx)
        df = pd.read_csv(Path(nbeats, i))
        real_values = concated_df[concated_df['label'] == idx]['value'].values
        targets[idx] = df['value']
        mases_f[idx] = MASE(df['value'], df['predict'], real_values)
global_mases[nbeats] = mases_f

mases_f = {}
for i in os.listdir(autogluon):
    if i.endswith('csv'):
        idx = i.split('_')[0]
        ids_to_proceed.append(idx)
        df = pd.read_csv(Path(autogluon, i))
        real_values = concated_df[concated_df['label'] == idx]['value'].values
        targets[idx] = df['value']
        mases_f[idx] = MASE(df['value'], df['predict'], real_values)
global_mases[autogluon] = mases_f

df_full = pd.DataFrame()

for k, v in global_mases.items():
    df = pd.DataFrame(v, index=[k])
    df_full = pd.concat([df_full, df], axis=0)

for i in ['D', 'W', 'Q', 'M', 'Y', 'all']:
    if i != 'all':
        filter_col = [col for col in df_full if col.startswith(i)]
        d__ = df_full[filter_col]
    else:
        d__ = df_full

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_df = pd.DataFrame()
    for q in quantiles:
        d_ = d__.dropna(axis=1)
        d___ = d_.quantile(q, axis=1).round(2)
        quantile_df = pd.concat([quantile_df, d___], axis=1)
        if np.isnan(d___[0]):
            continue
        print(i)
        print(f"{q}'s quantile ")
        print('MASE')
        for j, v in enumerate(d___.sort_values().index):
            print(j + 1, v, round(d___[v], 2))
        quantile_df.to_excel(f'quantiles_{i}.xlsx')
