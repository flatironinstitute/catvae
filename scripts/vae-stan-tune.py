import torch
import pystan
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from catvae.trainer import LightningCatVAE
import pickle


model_code = """
data {
  int<lower=0> N;      // number of samples
  int<lower=0> D;      // number of dimensions
  int<lower=0> K;      // number of latent dimensions
  matrix[D-1, D] Psi;  // Orthonormal basis
  int y[N, D];         // observed counts
}

parameters {
  // parameters required for linear regression on the species means
  matrix[N, D-1] eta; // ilr transformed abundances
  matrix[D-1, K] W;
  real<lower=0> sigma;

}

transformed parameters {
  matrix[D-1, D-1] Sigma;
  matrix[D-1, D-1] I;
  vector[D-1] z;
  I = diag_matrix(rep_vector(1.0, D-1));
  Sigma = W * W' + square(sigma) * I;
  z = rep_vector(0, D-1);
}

model {
  // generating counts
  for (n in 1:N){
     eta[n] ~ multi_normal(z, Sigma);
     y[n] ~ multinomial(softmax(to_vector(eta[n] * Psi)));
  }
}
"""


def main(args):
    if args.model == 'catvae':
        model = LightningCatVAE(args)
    elif args.model == 'linear-ae':
        model = LightningLinearVAE(args)
    else:
        raise ValueError(f'{args.model} is not supported')

    checkpoint = torch.load(
        args.torch_ckpt,
        map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    if args.stan_model is None:
        sm = pystan.StanModel(model_code=model_code)
    else:
        # load compiled model from pickle
        sm = pickle.load(open('model.pkl', 'rb'))
    W = model.model.decoder.weight.detach().cpu().numpy().squeeze()
    # b = model.model.decoder.bias.detach().cpu().numpy().squeeze()
    sigma = np.exp(0.5 * model.model.log_sigma_sq.detach().cpu().numpy())
    epochs = args.iterations // args.checkpoint_interval
    table = load_table(args.train_biom)
    N, D, K = table.shape[1], table.shape[0], args.n_latent
    psi = self.set_basis(N, table)
    Y = np.array(table.matrix_data.todense()).T
    fit_data = {'N': N, 'D': D, 'K': K, 'Psi': psi, 'y': Y}
    init = [{'W': W, 'sigma': sigma}] * args.chains
    fit = sm.sampling(data=fit_data, iter=args.iterations,
                      chains=args.chains, init=init)
    la = fit.extract(permuted=True)  # return a dictionary of arrays
    with open(f'{args.output_dir}/results.pkl', 'wb') as f:
        pickle.dump(la, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser = LightningCatVAE.add_model_specific_args(parser)
    parser.add_argument('--torch-ckpt', type=str, required=True,
                        help='Linear VAE checkpoint path.')
    parser.add_argument('--stan-model', type=str, default=None, required=False,
                        help='Path to compiled Stan model.')
    parser.add_argument('--model', type=str, default='catvae', required=False)
    parser.add_argument('--checkpoint-interval', type=int,
                        default=100, required=False,
                        help='Number of iterations per checkpoint.')
    parser.add_argument('--iterations', type=int,
                        default=1000, required=False,
                        help='Number of iterations.')
    parser.add_argument('--chains', type=int, default=4, required=False,
                        help='Number of MCMC chains to run in Stan')
    args = parser.parse_args()
    main(args)
