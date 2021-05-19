import unittest
import os
import shutil
import torch
from catvae.trainer import LightningLinearVAE, LightningBatchLinearVAE
from catvae.sim import multinomial_bioms, multinomial_batch_bioms
from biom import Table
from biom.util import biom_open
import numpy as np
from pytorch_lightning import Trainer
import argparse
import pandas as pd

from scipy.stats import pearsonr
from scipy.spatial.distance import pdist


class TestBatchVAEModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        k = 20
        C = 3
        self.sims = multinomial_batch_bioms(k=k, D=100, N=2000, M=1e6, C=C)
        Y = self.sims['Y']
        parts = Y.shape[0] // 10
        samp_ids = list(map(str, range(Y.shape[0])))
        obs_ids = list(map(str, range(Y.shape[1])))
        train = Table(Y[:parts * 8].T, obs_ids, samp_ids[:parts * 8])
        test = Table(Y[parts * 8: parts * 9].T,
                     obs_ids, samp_ids[parts * 8: parts * 9])
        valid = Table(Y[parts * 9:].T, obs_ids, samp_ids[parts * 9:])
        with biom_open('train.biom', 'w') as f:
            train.to_hdf5(f, 'train')
        with biom_open('test.biom', 'w') as f:
            test.to_hdf5(f, 'test')
        with biom_open('valid.biom', 'w') as f:
            valid.to_hdf5(f, 'valid')

        md = pd.DataFrame({'batch_category': self.sims['batch_idx']},
                          index=samp_ids)
        md.index.name = 'sampleid'
        md.to_csv('metadata.txt', sep='\t')
        batch_priors = pd.Series(self.sims['alphaILR'])
        batch_priors.to_csv('batch_priors.txt', sep='\t')
        self.sims['tree'].write('basis.nwk')

    def tearDown(self):
        os.remove('basis.nwk')
        os.remove('batch_priors.txt')
        os.remove('metadata.txt')
        os.remove('train.biom')
        os.remove('test.biom')
        os.remove('valid.biom')
        if os.path.exists('lightning_logs'):
            shutil.rmtree('lightning_logs')

    def test_fit(self):
        k = 20
        output_dir = 'output'
        args = [
            '--train-biom', 'train.biom',
            '--test-biom', 'test.biom',
            '--val-biom', 'valid.biom',
            '--sample-metadata', 'metadata.txt',
            '--batch-category', 'batch_category',
            '--batch-prior', 'batch_priors.txt',
            '--basis', 'basis.nwk',
            '--output-directory', output_dir,
            '--epochs', '50',
            '--batch-size', '200',
            '--encoder-depth', '1',
            '--num-workers', '3',
            '--scheduler', 'cosine',
            '--learning-rate', '1e-1',
            '--n-latent', f'{k}',
            '--n-hidden', '64',
            '--gpus', '0'
        ]

        parser = argparse.ArgumentParser(add_help=False)
        parser = LightningBatchLinearVAE.add_model_specific_args(parser)
        parser.add_argument('--num-workers', type=int)
        parser.add_argument('--gpus', type=int)
        args = parser.parse_args(args)
        model = LightningBatchLinearVAE(args)
        model.set_eigs(self.sims['eigvectors'], self.sims['eigs'])
        print(model)
        trainer = Trainer(
            max_epochs=args.epochs,
            gpus=args.gpus,
            check_val_every_n_epoch=1,
            # profiler=profiler,
            fast_dev_run=False,
            # auto_scale_batch_size='power'
        )
        trainer.fit(model)

        # See if the model can approximately recover W
        W = model.model.decoder.weight.detach().cpu().numpy()
        d_estW = pdist(W)
        simW = self.sims['W'] / np.sqrt(self.sims['eigs'])
        dW = pdist(simW)
        r, p = pearsonr(dW, d_estW)
        self.assertGreater(r, 0.5)
        self.assertLess(p, 1e-8)
        # See if the model can approximately remove beta
        B = model.model.beta.weight.detach().cpu().numpy().T
        d_estB = pdist(B)
        simB = self.sims['B'].T
        dB = pdist(simB)
        r, p = pearsonr(dB, d_estB)
        self.assertGreater(r, 0.3)
        self.assertLess(p, 1e-8)


class TestVAEModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        k = 10
        self.sims = multinomial_bioms(k=k, D=50, N=1000, M=50000)
        Y = self.sims['Y']
        parts = Y.shape[0] // 10
        samp_ids = list(map(str, range(Y.shape[0])))
        obs_ids = list(map(str, range(Y.shape[1])))
        train = Table(Y[:parts * 8].T, obs_ids, samp_ids[:parts * 8])
        test = Table(Y[parts * 8: parts * 9].T,
                     obs_ids, samp_ids[parts * 8: parts * 9])
        valid = Table(Y[parts * 9:].T, obs_ids, samp_ids[parts * 9:])
        with biom_open('train.biom', 'w') as f:
            train.to_hdf5(f, 'train')
        with biom_open('test.biom', 'w') as f:
            test.to_hdf5(f, 'test')
        with biom_open('valid.biom', 'w') as f:
            valid.to_hdf5(f, 'valid')
        self.sims['tree'].write('basis.nwk')

    def tearDown(self):
        os.remove('basis.nwk')
        os.remove('train.biom')
        os.remove('test.biom')
        os.remove('valid.biom')
        if os.path.exists('lightning_logs'):
            shutil.rmtree('lightning_logs')

    def test_run(self):
        output_dir = 'output'
        args = [
            '--train-biom', 'train.biom',
            '--test-biom', 'test.biom',
            '--val-biom', 'valid.biom',
            '--basis', 'basis.nwk',
            '--output-directory', output_dir,
            '--epochs', '50',
            '--batch-size', '50',
            '--num-workers', '10',
            '--scheduler', 'cosine',
            '--learning-rate', '1e-1',
            '--n-latent', '10',
            '--gpus', '0'
        ]
        parser = argparse.ArgumentParser(add_help=False)
        parser = LightningLinearVAE.add_model_specific_args(parser)
        parser.add_argument('--num-workers', type=int)
        parser.add_argument('--gpus', type=int)
        args = parser.parse_args(args)
        model = LightningLinearVAE(args)
        model.set_eigs(self.sims['eigvectors'], self.sims['eigs'])

        trainer = Trainer(
            max_epochs=args.epochs,
            gpus=args.gpus,
            check_val_every_n_epoch=10,
            fast_dev_run=False,
        )
        trainer.fit(model)

        # Make sure that the estimates are darn close
        W = model.model.decoder.weight.detach().cpu().numpy()
        d_estW = pdist(W)
        simW = self.sims['W'] / np.sqrt(self.sims['eigs'])
        dW = pdist(simW)
        r, p = pearsonr(dW, d_estW)
        self.assertGreater(r, 0.9)
        self.assertLess(p, 1e-8)


if __name__ == '__main__':
    unittest.main()
