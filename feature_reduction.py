'''
Usage: 
Please intall umap before.
Input: 
F(D_{real}), F(D_{syn}), latent_dim
Output:
real_emb, syn_emb, df
'''

import numpy as np 
import pandas as pd
import umap
import plotly.express as px
import os

def umap_embedding(real_data, syn_data, num_components=3, isReal2Syn=True, randn_state=42):
    '''
    @Input: 
    real_data: np arr, size = ()
    syn_data: np arr, size = ()
    num_components: default = 3
    isReal2Syn: specify if you want to map real data to syn latent space or the other way around.
    random_state: default = 42
    @Ouput:
    real_emb: np arr, size = 
    syn_emb: np arr, size = 
    '''
    if isReal2Syn:
        base, map_ = real_data, syn_data
    else:
        base, map_ = syn_data, real_data
    umap_model = umap.UMAP(random_state=randn_state, n_components=num_components).fit(base)
    base_emb = umap_model.transform(base)
    map_emb = umap_model.transform(map_)
    if isReal2Syn:
        real_emb, syn_emb = base_emb, map_emb
    else:
        real_emb, syn_emb = map_emb, base_emb
    return real_emb, syn_emb


def pca_embedding(real_data, syn_data, num_components=3, isReal2Syn=True, randn_state=42):
    pass    # complete with nmf
    return real_emb, syn_emb


def nmf_embedding(real_data, syn_data, num_components=3, isReal2Syn=True, randn_state=42):
    pass    # complete with nmf
    return real_emb, syn_emb


def getDF(real_emb, syn_emb, r_lst, s_lst):
    '''
    @Input: 
    real_emb: np arr 
    syn_emb: np arr
    r_lst: a list of real image files
    s_lst: a list of synthetic image files
    @Output: 
    df: pandas df of shape (1000,3) 
    where the first 500 images are real and last 5000 are syn.
    '''
    latent_dims = np.vstack([real_emb,syn_emb])
    type_col = np.asarray(['real']*500+['syn']*500).reshape(-1,1)
    file_col = np.array(r_lst+s_lst).reshape(-1,1)
    df_ = np.hstack([latent_dims, file_col, type_col])
    df = pd.DataFrame(df_)
    return df
