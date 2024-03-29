import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr


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
                raise ValueError(
                    "Layer type {} not supported!".format(type(layer)))
        return weight_tensor


def metric_transpose_theorem(model):
    """
    Metric for how close encoder and decoder.T are
    :param model: LinearAE model
    :return: ||W1 - W2^T||_F^2 / hidden_dim
    """
    # encoder_weight = get_weight_tensor_from_seq(model.encoder)
    # decoder_weight = get_weight_tensor_from_seq(model.decoder)
    encoder_weight = model.encoder.weight.cpu().numpy()
    decoder_weight = model.decoder.weight.cpu().numpy()
    transpose_metric = np.linalg.norm(encoder_weight - decoder_weight.T) ** 2
    return transpose_metric.item() / float(model.hidden_dim)


def metric_orthogonality(model):
    """ Measures how orthogonal the decoder matrix is. """
    W = model.decoder.weight.cpu().numpy()
    u, s, v = np.linalg.svd(W)
    eigvals = (W**2).sum(axis=0)
    Weig = W / np.sqrt(eigvals)
    Id = np.eye(Weig.shape[1])
    eigvals = np.sqrt(np.sort(eigvals)[::-1])
    hd = float(model.hidden_dim)
    ortho_err = np.linalg.norm(Weig.T @ Weig - Id) ** 2 / hd
    if len(s) < len(eigvals):
        eigvals = eigvals[:len(s)]
    eig_err = np.sum((s - eigvals) ** 2) / hd
    return ortho_err, eig_err


def metric_alignment(model, gt_eigvectors):
    """
    Metric for alignment of decoder columns to ground truth eigenvectors
    :param model: Linear AE model
    :param gt_eigvectors: ground truth eigenvectors (input_dims,hidden_dims)
    :return: sum_i (1 - max_j (cos(eigvector_i, normalized_decoder column_j)))
    """
    decoder_np = model.decoder.weight.cpu().numpy()[:gt_eigvectors.shape[0], :]

    # normalize columns of gt_eigvectors
    norm_gt_eigvectors = gt_eigvectors / np.linalg.norm(gt_eigvectors, axis=0)
    # normalize columns of decoder
    norm_decoder = decoder_np / (np.linalg.norm(decoder_np, axis=0) + 1e-8)

    total_angles = 0.0
    for eig_i in range(gt_eigvectors.shape[1]):
        eigvector = norm_gt_eigvectors[:, eig_i]
        total_angles += 1. - np.max(np.abs(norm_decoder.T @ eigvector)) ** 2

    return total_angles / float(model.hidden_dim)


def metric_subspace(model, gt_eigvectors, gt_eigs):
    decoder_np = model.decoder.weight.cpu().numpy()[:gt_eigvectors.shape[0], :]

    # if correction:
    #     C = model.Psi.to_dense()
    #     A = torch.svd(model.encode(C))[1].cpu()
    #     A = torch.diag(A).detach().numpy()
    #     decoder_np = decoder_np @ A

    # k - tr(UU^T WW^T), where W is left singular vector matrix of decoder
    u, s, vh = np.linalg.svd(decoder_np, full_matrices=False)
    hd = float(model.hidden_dim)
    return 1 - np.trace(gt_eigvectors @ gt_eigvectors.T @ u @ u.T) / hd


def metric_procrustes(model, gt_eigvectors):
    W = model.decoder.weight.cpu().numpy()[:gt_eigvectors.shape[0], :]
    return procrustes(gt_eigvectors, W)[2]


def metric_pairwise(model, gt_eigvectors, gt_eigs):
    W = model.decoder.weight.cpu().numpy()[:gt_eigvectors.shape[0], :]
    trueW = gt_eigvectors @ np.diag(gt_eigs)
    true_dW = pdist(trueW)
    est_dW = pdist(W)
    r, _ = pearsonr(true_dW, est_dW)
    return r
