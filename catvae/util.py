import os
import yaml
import torch
import skbio
import numpy as np
import pandas as pd
from catvae.trainer import MultVAE
from gneiss.balances import sparse_balance_basis


def load_model(model_path):
    """ Loads VAE model.

    Parameters
    ----------
    model_path : str
       Path to the pretrained VAE model

    Returns
    ----------
    vae_model : MultVAE
        Pretrained Multinomial VAE
    tree : skbio.TreeNode
        The tree used to train the VAE
    """
    ckpt_path = os.path.join(model_path, 'last_ckpt.pt')
    params = os.path.join(model_path, 'hparams.yaml')
    nwk_path = os.path.join(model_path, 'tree.nwk')
    tree = skbio.TreeNode(nwk_path)
    with open(params, 'r') as stream:
        params = yaml.safe_load(stream)
    params['basis'] = nwk_path
    vae_model = MultVAE.load_from_checkpoint(ckpt_path, **params)
    return vae_model, tree


def extract_sample_embeddings(vae_model, tree, table, return_type='dataframe'):
    """ Extracts sample embeddings from model

    Parameters
    ----------
    vae_model : MultVAE
        Pretrained Multinomial VAE
    tree : skbio.TreeNode
        The tree used to train the VAE
    table : biom.Table
        The biom table one wishes to convert to sample embeddings
    return_type : str
        Options include 'tensor', 'array', 'dataframe' (default='tensor').
        If 'tensor' is specified, a `torch.Tensor` object is returned.
        If 'array' is specified, a `numpy.array` object is returned.
        If 'dataframe' is specified, a `pandas.DataFrame` object is returned.
    """
    X = X.to_dataframe().to_dense()
    tips = [n.name for n in tree.tips()]
    X = X.reindex(index=tips).fillna(0)
    X_embed = vae_model.to_latent(
        torch.Tensor(X.values.T).float())
    if return_type == 'tensor':
        return X_embed
    X_embed = X_embed.detach().cpu().numpy()
    if return_type == 'array':
        return X_embed
    elif return_type == 'dataframe':
        return pd.DataFrame(X_embed, index=table.ids(axis='sample'))
    else:
        ValueError(f'return type {return_type} is not supported.')


def extract_observation_embeddings(vae_model, tree, return_type='dataframe'):
    """ Extracts observation embeddings from model (i.e. OTUs).

    The observation embeddings are all represented in CLR coordinates.

    Parameters
    ----------
    vae_model : MultVAE
        Pretrained Multinomial VAE
    tree : skbio.TreeNode
        The tree used to train the VAE
    return_type : str
        Options include 'tensor', 'array', 'dataframe' (default='dataframe')
    """
    # ILR representation of the VAE decoder loadings
    W = vae_model.vae.decoder.weight
    Psi, _ = sparse_balance_basis(tree)
    if return_type == 'torch':
        indices = np.vstack((Psi.row, Psi.col))
        Psi = torch.sparse_coo_tensor(
            indices.copy(), Psi.data.astype(np.float32).copy(),
            requires_grad=False).coalesce()
        return Psi.T @ W
    if return_type == 'array':
        return Psi.T @ W.detach().numpy()
    if return_type == 'dataframe':
        names = [n.name for n in tree.tips()]
        return pd.DataFrame(Psi.T @ W.detach().numpy(), index=names)
    else:
        ValueError(f'return type {return_type} is not supported.')
