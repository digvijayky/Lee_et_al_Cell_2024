#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import scipy as sp
import scanpy as sc
import seaborn as sns
import h5py
import pandas as pd
from matplotlib import pyplot as plt, colors, rc
from pdb import set_trace

rc('font',**{'family':'serif','serif':['Arial']})
rc('figure',**{'dpi':200,})

data_dir = "path/to/ashelyetal/data/"

filename = "PATIENT_LUNG_ADENOCARCINOMA_ANNOTATED.h5"
dataset = 'INDF_EPITHELIAL_NOR_TUMOR_MET'
df = pd.read_hdf(f'{data_dir}/{filename}', key=dataset).reset_index()
temp = df['DEVELOPMENT_PHENOGRAPH_CLASS'] # this is in the correct order already
df = df.sort_values('LUNG_EPITHELIUM_RANK')
df['DEVELOPMENT_PHENOGRAPH_CLASS'] = temp.values
df = df[df['Meta-Source']=='MET']
df = df[df['DEVELOPMENT_PHENOGRAPH_CLASS'].isin(['II', 'III'])]

with open(f'{data_dir}/cell_cycle.regev_etal.gmt') as f:
    cell_cycle = [[g.strip('\n') for g in gs.split("\t")] for gs in f.readlines()]
    cell_cycle = [g for gs in cell_cycle for g in gs[2:]]
    cell_cycle = df.columns.intersection(cell_cycle)

filename = "stem_cell_signatures_merged.csv"
hallmark_emt = pd.read_csv(f"{data_dir}/{filename}")['HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION'].dropna()
hallmark_emt = df.columns.intersection(hallmark_emt).difference(['CCN1', 'CCN2'])

# features to plot
genes_to_plot = {'EMT TFs': ['SNAI1', 'SNAI2', 'ZEB1', 'ZEB2'], 'Fibrogenic factors': ['IL11', 'PDGFB', 'HAS2', 'WISP1', 'CTGF', 'SERPINE1', 'CCBE1', 'COL6A1', 'LAMA3'], 'TGF-ß \n isoforms': ['TGFB1', 'TGFB2', 'TGFB3'], 'EMT\nsig': ['Fibrogenic EMT', 'EMT TF sig', 'Hallmark EMT']}

# Score gene signatures
df['Proliferation'] = np.nanmean(df[cell_cycle], axis=1)
df['Hallmark EMT'] = np.nanmean(df[hallmark_emt], axis=1)
df['Fibrogenic EMT'] = np.nanmean(df[genes_to_plot['Fibrogenic factors']], axis=1)
df['EMT TF sig'] = np.nanmean(df[genes_to_plot['EMT TFs']], axis=1)
sort_param = ['Hallmark EMT']

features = np.concatenate(list(genes_to_plot.values()))

# Get smoothed matrix to plot
mat = df[features].apply(sp.stats.zscore, axis=0)
mat = mat.sort_values(by=sort_param, ascending=False)
df.to_hdf('/data/massague/vijay/scrna_analysisg/zhenghans_project/datasets/laugney_et_al/ashley_nature_med/data/ashley_patient_luad_data_with_hallmark_emt_sig.h5', 'df')
df = df.loc[mat.index, :]

# Get categorical data to plot
col = 'DEVELOPMENT_PHENOGRAPH_CLASS'
class_colors = {'I-P': '#F0F0F0', 'I-Q': '#E0E0E0', 'II': '#A0A0A0', 'III': '#202020'}
vals, idx = pd.factorize(df[col])
cmap = colors.ListedColormap(idx.map(class_colors))

fig, (ax1, ax2) = plt.subplots( 2, 1, figsize = (10,10), gridspec_kw={'height_ratios': [1, 30], 'hspace':2e-2})
ax1.axis("Off")
sns.heatmap( mat.T, cmap="RdBu_r", vmin=-2, vmax=2, cbar=True, ax=ax2)
ax2.xaxis.set_visible(False)
ax2.tick_params(width=0)
fig.suptitle('Expression of fibrogenic EMT genes (jing et al version of ashley et al data)', fontsize=16)
fig.savefig(os.path.join('/data/massague/vijay/scrna_analysisg/zhenghans_project/datasets/laugney_et_al/', 'figures/regenerative_dataset_fibrogenic_emt_orig_code.pdf'))
