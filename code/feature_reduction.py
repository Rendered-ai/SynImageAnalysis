import numpy as np
import pandas as pd
import umap


def umap_embedding(real_data, syn_data, num_components=3, isReal2Syn=True, randn_state=42):
    """
    Map real data to synthetic data's latent space.

    :param real_data: A flattened array in shape (n_images, 512*512*3).
    :param syn_data: A flattened array in shape (
    n_images, 512*512*3).
    :param num_components: by default = 3
    :param isReal2Syn: Bool value. If True, mapping real
    data to synthetic data's latent space. Else, mapping synthetic data to real data's latent space.
    :param randn_state: random seed
    :return:
    real_emb: An array of shape (n_images, num_components)
    syn_emb: An array of shape (n_images, num_components)
    """
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


def getDF(real_emb, syn_emb, r_lst, s_lst):
    """
    Create a df for reduced-shape synthetic and real data.

    :param real_emb: An array of shape (n_images, latent_dim)
    :param syn_emb: An array of shape (n_images, latent_dim)
    :param r_lst: A list of real_image_filenames. len = n_images.
    :param s_lst: A list of synthetic_image_filenames. len = n_images.
    :return: A pandas dataframe of shape (2 * n_images, latent_dim + 2) with columns:
    latent dim 1, latent dim2, latent dim3, latent dim4, ..., (if there are any), filename, data_type.
    where data_type is either synthetic or real.
    """
    latent_dims = np.vstack([real_emb,syn_emb])
    n = len(r_lst)
    type_col = np.asarray(['real'] * n + ['syn'] * n).reshape(-1, 1)
    file_col = np.array(r_lst + s_lst).reshape(-1, 1)
    df_ = np.hstack([latent_dims, file_col, type_col])
    df = pd.DataFrame(df_)
    return df
