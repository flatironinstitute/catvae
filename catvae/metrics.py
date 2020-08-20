import torch
import torch.nn as nn
import numpy as np


def get_weight_tensor_from_seq(weight_seq):
    if isinstance(weight_seq, nn.Linear):
        return weight_seq.weight.detach()
    elif isinstance(weight_seq, nn.Sequential):
        weight_tensor = None
        for layer in weight_seq:
            if isinstance(layer, nn.Linear):
                layer_weight = layer.weight.detach()
                if weight_tensor is None:
                    weight_tensor = layer_weight
                else:
                    weight_tensor = layer_weight @ weight_tensor
            elif isinstance(layer, nn.BatchNorm1d):
                bn_weight = layer.weight.detach()

                # ignore bias

                if weight_tensor is None:
                    weight_tensor = torch.diag(bn_weight)
                else:
                    weight_tensor = torch.diag(bn_weight) @ weight_tensor
            else:
                raise ValueError("Layer type {} not supported!".format(type(layer)))
        return weight_tensor


def metric_transpose_theorem(model):
    """
    Metric for how close encoder and decoder.T are
    :param model: LinearAE model
    :return: ||W1 - W2^T||_F^2 / hidden_dim
    """
    # encoder_weight = get_weight_tensor_from_seq(model.encoder)
    # decoder_weight = get_weight_tensor_from_seq(model.decoder)
    encoder_weight = model.get_loadings(decoder=False)
    decoder_weight = model.get_loadings(decoder=True)
    transpose_metric = np.linalg.norm(encoder_weight - decoder_weight.T) ** 2
    return transpose_metric.item() / float(model.n_latent)


def metric_alignment(model, gt_eigvectors):
    """
    Metric for alignment of decoder columns to ground truth eigenvectors
    :param model: Linear AE model
    :param gt_eigvectors: ground truth eigenvectors (input_dims,hidden_dims)
    :return: sum_i (1 - max_j (cos(eigvector_i, normalized_decoder column_j)))
    """
    #decoder_weight = get_weight_tensor_from_seq(model.decoder)
    decoder_np = model.get_loadings(decoder=True)

    # normalize columns of gt_eigvectors
    norm_gt_eigvectors = gt_eigvectors / np.linalg.norm(gt_eigvectors, axis=0)
    # normalize columns of decoder
    norm_decoder = decoder_np / (np.linalg.norm(decoder_np, axis=0) + 1e-8)

    total_angles = 0.0
    for eig_i in range(gt_eigvectors.shape[1]):
        eigvector = norm_gt_eigvectors[:, eig_i]
        total_angles += 1. - np.max(np.abs(norm_decoder.T @ eigvector)) ** 2

    return total_angles / float(model.n_latent)


def metric_subspace(model, gt_eigvectors, gt_eigs):
    #decoder_weight = get_weight_tensor_from_seq(model.decoder)
    decoder_np = model.get_loadings(decoder=True)

    # k - tr(UU^T WW^T), where W is left singular vector matrix of decoder
    u, s, vh = np.linalg.svd(decoder_np, full_matrices=False)
    return 1 - np.trace(gt_eigvectors @ gt_eigvectors.T @ u @ u.T) / float(model.n_latent)
