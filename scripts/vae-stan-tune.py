import os
import torch
import pystan
import numpy as np
import argparse
from catvae.trainer import MultVAE, BiomDataModule, add_data_specific_args
from catvae.composition import ilr_basis
import pickle
from biom import load_table


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

def set_basis(n_input, basis):
    # a sneak peek into file types to initialize model
    basis = ilr_basis(basis)
    assert basis.shape[1] == n_input, (
        f'Basis shape {basis.shape} does '
        f'not match tree dimension {n_input}. '
        'Also make sure if your tree if aligned correctly '
        'with `gneiss.util.match_tips`')
    return np.array(basis.todense())


def main(args):

    table = load_table(args.train_biom)
    N, D, K = table.shape[1], table.shape[0], args.n_latent
    psi = set_basis(D, basis=args.basis)
    Y = np.array(table.matrix_data.todense()).T.astype(np.int64)
    fit_data = {'N': N, 'D': D, 'K': K, 'Psi': psi, 'y': Y}
    if args.model == 'linear-vae':
        model = MultVAE(args)
        checkpoint = torch.load(
            args.torch_ckpt,
            map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

        W = model.model.decoder.weight.detach().cpu().numpy().squeeze()
        sigma = np.exp(0.5 * model.model.log_sigma_sq.detach().cpu().numpy())
        init = [{'W': W, 'sigma': sigma}] * args.chains

    else:
        init = None

    if args.stan_model is None:
        sm = pystan.StanModel(model_code=model_code)
    else:
        if os.path.exists(args.stan_model):
            # load compiled model from pickle
            sm = pickle.load(open(args.stan_model, 'rb'))
        else:
            sm = pystan.StanModel(model_code=model_code)
            with open(args.stan_model, 'wb') as f:
                pickle.dump(sm, f)

    if args.mode == 'hmc':
        fit = sm.sampling(data=fit_data, iter=args.iterations,
                          chains=args.chains)
        print('Sampling from posterior distribution')
        # return a dictionary of arrays
        la = fit.extract(permuted=False, pars=['W', 'sigma'])
        print('Saving file')
        with open(f'{args.output_directory}/hmc-results.pkl', 'wb') as f:
            pickle.dump(la, f)
    elif args.mode == 'mle':
        la = sm.optimizing(data=fit_data, iter=args.iterations)
        with open(f'{args.output_directory}/mle-results.pkl', 'wb') as f:
            pickle.dump(la, f)
    else:
        raise ValueError(f'{args.mode} not implemented.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser = MultVAE.add_model_specific_args(parser)
    parser = add_data_specific_args(parser, add_help=False)
    parser.add_argument('--torch-ckpt', type=str, required=False,
                        help='Linear VAE checkpoint path.')
    parser.add_argument('--stan-model', type=str, default=None, required=False,
                        help='Path to compiled Stan model.')
    parser.add_argument('--model', type=str, default='linear-vae', required=False)
    parser.add_argument('--checkpoint-interval', type=int,
                        default=100, required=False,
                        help='Number of iterations per checkpoint.')
    parser.add_argument('--iterations', type=int,
                        default=1000, required=False,
                        help='Number of iterations.')
    parser.add_argument('--mode', type=str,
                        default='hmc', required=False,
                        help='Specifies either `hmc` or `mle`.')
    parser.add_argument('--chains', type=int, default=4, required=False,
                        help='Number of MCMC chains to run in Stan')
    args = parser.parse_args()
    main(args)
