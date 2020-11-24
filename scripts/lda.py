import os
import torch
import pystan
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pickle
from biom import load_table



def main(args):
    model = LDA(n_components=args.n_latent, max_iter=args.iterations,
                verbose=1, learning_method='online')
    table = load_table(args.train_biom)
    X = table.matrix_data.T
    model.fit(X)
    with open(args.model_checkpoint, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-biom', help='Training biom file', required=True)
    parser.add_argument('--n-latent', type=int, help='Number of components')
    parser.add_argument('--iterations', type=int,
                        default=10000, required=False,
                        help='Number of iterations.')
    parser.add_argument('--batch-size', type=int,
                        default=256, required=False,
                        help='Batch size')
    parser.add_argument('--n-jobs', type=int,
                        default=-1, required=False,
                        help='Number of concurrent jobs.')
    parser.add_argument('--model-checkpoint',
                        required=True,
                        help='Location of saved model.')
    args = parser.parse_args()
    main(args)
