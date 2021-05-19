import argparse
import numpy as np
from biom import Table
from catvae.sim import multinomial_bioms
from biom.util import biom_open
import os


def main(args):
    os.mkdir(args.output_dir)
    np.random.seed(args.seed)
    sims = multinomial_bioms(
        k=args.latent_dim, D=args.input_dim,
        N=args.samples, M=args.depth)
    Y = sims['Y']
    parts = Y.shape[0] // 10
    samp_ids = list(map(str, range(Y.shape[0])))
    obs_ids = list(map(str, range(Y.shape[1])))
    train = Table(Y[:parts * 8].T, obs_ids, samp_ids[:parts * 8])
    test = Table(Y[parts * 8: parts * 9].T,
                 obs_ids, samp_ids[parts * 8: parts * 9])
    valid = Table(Y[parts * 9:].T, obs_ids, samp_ids[parts * 9:])
    output_dir = args.output_dir
    with biom_open(f'{output_dir}/train.biom', 'w') as f:
        train.to_hdf5(f, 'train')
    with biom_open(f'{output_dir}/test.biom', 'w') as f:
        test.to_hdf5(f, 'test')
    with biom_open(f'{output_dir}/valid.biom', 'w') as f:
        valid.to_hdf5(f, 'valid')
    tree = sims['tree']
    tree.write(f'{output_dir}/basis.nwk')
    np.savetxt(f'{output_dir}/eigvals.txt', sims['eigs'])
    np.savetxt(f'{output_dir}/eigvecs.txt', sims['eigvectors'])
    np.savetxt(f'{output_dir}/W.txt', sims['W'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--latent-dim', type=int, default=10)
    parser.add_argument('--input-dim', type=int, default=100)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--depth', type=int, default=10000)
    parser.add_argument('--output-dir', type=str)
    args = parser.parse_args()
    main(args)
