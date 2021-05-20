import unittest
import os
import shutil
import torch
from catvae.trainer import MultVAE, MultBatchVAE, BiomDataModule
from catvae.sim import multinomial_bioms, multinomial_batch_bioms
from biom import Table
from biom.util import biom_open
import numpy as np
from pytorch_lightning import Trainer
import pandas as pd

from scipy.stats import pearsonr
from scipy.spatial.distance import pdist


class TestVAEModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        torch.manual_seed(1)
        self.k, self.D, self.N, self.M = 10, 50, 500, 100000
        self.sims = multinomial_bioms(k=self.k, D=self.D,
                                      N=self.N, M=self.M)
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
        if os.path.exists('lightning_logs'):
            shutil.rmtree('lightning_logs')
        if os.path.exists('summary'):
            shutil.rmtree('summary')
        os.remove('basis.nwk')
        os.remove('train.biom')
        os.remove('test.biom')
        os.remove('valid.biom')

    def test_run(self):
        model = MultVAE(n_input=self.D, n_latent=self.k,
                        n_hidden=16,  basis='basis.nwk',
                        dropout=0.5, bias=True, batch_norm=True,
                        encoder_depth=1, learning_rate=0.1,
                        scheduler='cosine', transform='pseudocount')
        model.set_eigs(self.sims['eigvectors'], self.sims['eigs'])
        dm = BiomDataModule('train.biom', 'test.biom', 'valid.biom',
                            batch_size=50)
        trainer = Trainer(
            max_epochs=50,
            gpus=0,
            check_val_every_n_epoch=10,
            fast_dev_run=False,
        )
        trainer.fit(model, dm)

        # Make sure that the estimates are darn close
        W = model.vae.decoder.weight.detach().cpu().numpy()
        d_estW = pdist(W)
        simW = self.sims['W'] / np.sqrt(self.sims['eigs'])
        dW = pdist(simW)
        r, p = pearsonr(dW, d_estW)
        self.assertGreater(r, 0.9)
        self.assertLess(p, 1e-8)


class TestBatchVAEModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)
        self.k, self.D, self.N, self.M, self.C = 10, 50, 500, 100000, 3
        self.sims = multinomial_batch_bioms(k=self.k, D=self.D,
                                            N=self.N, M=self.M, C=self.C)
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
        model = MultBatchVAE(n_input=self.D, n_latent=self.k,
                             n_hidden=16,  n_batches=self.C,
                             basis='basis.nwk', batch_prior='batch_priors.txt',
                             dropout=0.5, bias=True, batch_norm=True,
                             encoder_depth=1, learning_rate=0.1,
                             scheduler='cosine', transform='pseudocount')

        model.set_eigs(self.sims['eigvectors'], self.sims['eigs'])
        print(model)
        dm = BiomDataModule('train.biom', 'test.biom', 'valid.biom',
                            metadata='metadata.txt',
                            batch_category='batch_category',
                            batch_size=50)

        trainer = Trainer(
            max_epochs=50,
            gpus=0,
            check_val_every_n_epoch=1,
            fast_dev_run=False,
        )
        trainer.fit(model, dm)

        # See if the model can approximately recover W
        W = model.vae.decoder.weight.detach().cpu().numpy()
        d_estW = pdist(W)
        simW = self.sims['W'] / np.sqrt(self.sims['eigs'])
        dW = pdist(simW)
        r, p = pearsonr(dW, d_estW)
        self.assertGreater(r, 0.5)
        self.assertLess(p, 1e-8)
        # See if the model can approximately remove beta
        B = model.vae.beta.weight.detach().cpu().numpy().T
        d_estB = pdist(B)
        simB = self.sims['B'].T
        dB = pdist(simB)
        r, p = pearsonr(dB, d_estB)
        self.assertGreater(r, 0.3)
        self.assertLess(p, 1e-8)


if __name__ == '__main__':
    unittest.main()
