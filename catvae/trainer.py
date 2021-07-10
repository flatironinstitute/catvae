import datetime
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, StepLR,
    CosineAnnealingLR)
from catvae.dataset.biom import (
    BiomDataset, TripletDataset,
    collate_single_f, collate_q2_triplet_f,
    collate_batch_f, )
from catvae.models import LinearVAE, LinearBatchVAE, TripletNet
from catvae.composition import (alr_basis, ilr_basis, identity_basis, closure)
from catvae.metrics import (
    metric_subspace, metric_pairwise,
    metric_procrustes, metric_alignment, metric_orthogonality)
import pytorch_lightning as pl

from biom import load_table
import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
import os


class BiomDataModule(pl.LightningDataModule):
    def __init__(self, train_biom, test_biom, valid_biom,
                 metadata=None, batch_category=None,
                 batch_size=10, num_workers=1):
        super().__init__()
        self.train_biom = train_biom
        self.test_biom = test_biom
        self.val_biom = valid_biom
        self.batch_size = batch_size
        self.num_workers = num_workers
        if metadata is not None:
            self.metadata = pd.read_table(
                metadata, dtype=str)
            index_name = self.metadata.columns[0]
            self.metadata = self.metadata.set_index(index_name)
        else:
            self.metadata = None
        self.batch_category = batch_category
        if self.batch_category is None:
            self.collate_f = collate_single_f
        else:
            self.collate_f = collate_batch_f
        # collect class mappings if they exist
        if batch_category is not None:
            train_dataset = BiomDataset(
                load_table(self.train_biom),
                metadata=self.metadata, batch_category=self.batch_category)
            self.batch_categories = train_dataset.batch_cats

    def train_dataloader(self):
        train_dataset = BiomDataset(
            load_table(self.train_biom),
            metadata=self.metadata, batch_category=self.batch_category)
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            collate_fn=self.collate_f, shuffle=True,
            num_workers=self.num_workers, drop_last=True,
            pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = BiomDataset(
            load_table(self.val_biom),
            metadata=self.metadata, batch_category=self.batch_category)
        batch_size = min(len(val_dataset) - 1, self.batch_size)
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size,
            collate_fn=self.collate_f, shuffle=False,
            num_workers=self.num_workers, drop_last=True,
            pin_memory=True)
        return val_dataloader

    def test_dataloader(self):
        test_dataset = BiomDataset(
            load_table(self.test_biom),
            metadata=self.metadata, batch_category=self.batch_category)
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size,
            collate_fn=self.collate_f,
            shuffle=False, num_workers=self.num_workers,
            drop_last=True, pin_memory=True)
        return test_dataloader


class TripletDataModule(pl.LightningDataModule):
    def __init__(self, train_biom, test_biom, valid_biom,
                 metadata, batch_category, class_category,
                 batch_size=10, num_workers=1):
        super().__init__()
        self.train_biom = train_biom
        self.test_biom = test_biom
        self.val_biom = valid_biom
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_category = batch_category
        self.class_category = class_category
        self.metadata = pd.read_table(
            metadata, dtype=str)
        index_name = self.metadata.columns[0]
        self.metadata = self.metadata.set_index(index_name)

    def train_dataloader(self):
        train_dataset = TripletDataset(
            load_table(self.train_biom),
            metadata=self.metadata,
            batch_category=self.batch_category,
            class_category=self.class_category)
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            collate_fn=collate_q2_triplet_f, shuffle=True,
            num_workers=self.num_workers, drop_last=True,
            pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = TripletDataset(
            load_table(self.val_biom),
            metadata=self.metadata,
            batch_category=self.batch_category,
            class_category=self.class_category)
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size,
            collate_fn=collate_q2_triplet_f, shuffle=False,
            num_workers=self.num_workers, drop_last=True,
            pin_memory=True)
        return val_dataloader

    def test_dataloader(self):
        test_dataset = TripletDataset(
            load_table(self.test_biom),
            metadata=self.metadata,
            batch_category=self.batch_category,
            class_category=self.class_category)
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size,
            collate_fn=collate_q2_triplet_f,
            shuffle=False, num_workers=self.num_workers,
            drop_last=True, pin_memory=True)
        return test_dataloader


class MultVAE(pl.LightningModule):
    def __init__(self, n_input, n_latent=32, n_hidden=64, basis=None,
                 dropout=0.5, bias=True, tss=False, batch_norm=False,
                 encoder_depth=1, learning_rate=0.001, scheduler='cosine',
                 transform='pseudocount', distribution='multinomial',
                 grassmannian=True):
        super().__init__()
        # a hack to avoid the save_hyperparameters anti-pattern
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/7443
        self._hparams = {
            'n_input': n_input,
            'n_latent': n_latent,
            'n_hidden': n_hidden,
            'basis': basis,
            'dropout': dropout,
            'bias': bias,
            'tss' : tss,
            'batch_norm': batch_norm,
            'encoder_depth': encoder_depth,
            'learning_rate': learning_rate,
            'scheduler': scheduler,
            'transform': transform,
            'distribution': distribution,
            'grassmannian': grassmannian,
        }
        basis = self.set_basis(n_input, basis)
        self.vae = LinearVAE(
            n_input, basis=basis,
            hidden_dim=n_hidden,
            latent_dim=n_latent,
            bias=bias,
            encoder_depth=encoder_depth,
            batch_norm=batch_norm,
            dropout=dropout,
            distribution=distribution,
            transform=transform,
            grassmannian=grassmannian)
        self.gt_eigvectors = None
        self.gt_eigs = None

    def set_basis(self, n_input, basis=None):
        # a sneak peek into file types to initialize model
        has_basis = basis is not None
        if (has_basis and os.path.exists(basis)):
            basis = ilr_basis(basis)
            assert basis.shape[1] == n_input, (
                f'Basis shape {basis.shape} does '
                f'not match tree dimension {n_input}. '
                'Also make sure if your tree if aligned correctly '
                'with `gneiss.util.match_tips`')
        elif basis == 'alr':
            basis = coo_matrix(alr_basis(n_input))
        elif basis == 'identity':
            basis = coo_matrix(identity_basis(n_input))
        else:
            basis = None
        return basis

    def set_eigs(self, gt_eigvectors, gt_eigs):
        self.gt_eigvectors = gt_eigvectors
        self.gt_eigs = gt_eigs

    def forward(self, X):
        return self.vae(X)

    def to_latent(self, X):
        return self.vae.encode(X)

    def initialize_logging(self, root_dir='./', logging_path=None):
        if logging_path is None:
            basename = "logdir"
            suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            logging_path = "_".join([basename, suffix])
        full_path = root_dir + logging_path
        writer = SummaryWriter(full_path)
        return writer

    def training_step(self, batch, batch_idx):
        self.vae.train()
        counts = batch.to(self.device)
        if self.hparams['tss']:  # only for benchmarking
            counts = closure(counts)
        loss = self.vae(counts)
        assert torch.isnan(loss).item() is False
        if len(self.trainer.lr_schedulers) >= 1:
            lr = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
            current_lr = lr
        else:
            current_lr = self.hparams['learning_rate']
        tensorboard_logs = {
            'train_loss': loss, 'elbo': -loss, 'lr': current_lr
        }
        # log the learning rate
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            counts = batch
            if self.hparams['tss']:  # only for benchmarking
                counts = closure(counts)
            loss = self.vae(counts)
            assert torch.isnan(loss).item() is False

            # Record the actual loss.
            rec_err = self.vae.get_reconstruction_loss(batch)
            tensorboard_logs = {'val_loss': loss,
                                'val_rec_err': rec_err}
            # log the learning rate
            return {'val_loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        loss_f = lambda x: x['log']['val_rec_err']
        losses = list(map(loss_f, outputs))
        rec_err = sum(losses) / len(losses)
        self.logger.experiment.add_scalar('val_rec_err',
                                          rec_err, self.global_step)

        loss_f = lambda x: x['log']['val_loss']
        losses = list(map(loss_f, outputs))
        loss = sum(losses) / len(losses)
        self.logger.experiment.add_scalar('val_loss',
                                          loss, self.global_step)
        self.log('val_loss', loss)

        # Commenting out, since it is too slow...
        # ortho, eig_err = metric_orthogonality(self.vae)
        # self.logger.experiment.add_scalar('orthogonality',
        #                                   ortho, self.global_step)
        # tensorboard_logs = dict(
        #     [('val_loss', loss),
        #      ('orthogonality', ortho),
        #      ('eigenvalue-error', eig_err)]
        # )
        tensorboard_logs = {'val_loss': loss, 'val_rec_error': rec_err}

        if (self.gt_eigvectors is not None) and (self.gt_eigs is not None):
            ms = metric_subspace(self.vae, self.gt_eigvectors, self.gt_eigs)
            ma = metric_alignment(self.vae, self.gt_eigvectors)
            mp = metric_procrustes(self.vae, self.gt_eigvectors)
            mr = metric_pairwise(self.vae, self.gt_eigvectors, self.gt_eigs)
            tlog = {'subspace_distance': ms, 'alignment': ma, 'procrustes': mp}
            self.logger.experiment.add_scalar(
                'procrustes', mp, self.global_step)
            self.logger.experiment.add_scalar(
                'pairwise_r', mr, self.global_step)
            self.logger.experiment.add_scalar(
                'subspace_distance', ms, self.global_step)
            self.logger.experiment.add_scalar(
                'alignment', ma, self.global_step)
            tensorboard_logs = {**tensorboard_logs, **tlog}

        return {'val_loss': loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.vae.parameters(), lr=self.hparams['learning_rate'])
        if self.hparams['scheduler'] == 'cosine_warm':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=2, T_mult=2)
        elif self.hparams['scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer, T_max=10)
        elif self.hparams['scheduler'] == 'steplr':
            scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        elif self.hparams['scheduler'] == 'none':
            return [optimizer]
        else:
            s = self.scheduler
            raise ValueError(f'{s} is not implemented.')
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser, add_help=True):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=add_help)
        parser.add_argument(
            '--basis',
            help=('Basis.  Options include `alr`, `identity` or a file. '
                  'If a file is specified, it is assumed to be in '
                  'Newick file to specify basis from bifurcating tree. '
                  'If not specified, a random basis will be defined.'),
            required=False, type=str, default=None)
        parser.add_argument(
            '--n-latent', help='Latent embedding dimension.',
            required=False, type=int, default=10)
        parser.add_argument(
            '--n-hidden', help='Encoder dimension.',
            required=False, type=int, default=64)
        parser.add_argument(
            '--dropout', help='Dropout probability',
            required=False, type=float, default=0.1)
        parser.add_argument('--bias', dest='bias', action='store_true')
        parser.add_argument('--no-bias', dest='bias', action='store_false')
        #https://stackoverflow.com/a/15008806/1167475
        parser.set_defaults(bias=True)
        parser.add_argument('--tss', dest='tss', action='store_true',
                            help=('Total sum scaling to convert counts '
                                  'to proportions.  This option is highly '
                                  'recommended against and will not be '
                                  'supported in the future.'))
        parser.set_defaults(tss=False)
        parser.add_argument('--batch-norm', dest='batch_norm',
                            action='store_true')
        parser.add_argument('--no-batch-norm', dest='batch_norm',
                            action='store_false')
        parser.set_defaults(batch_norm=True)
        parser.add_argument(
            '--encoder-depth', help='Number of encoding layers.',
            required=False, type=int, default=1)
        parser.add_argument(
            '--learning-rate', help='Learning rate',
            required=False, type=float, default=1e-3)
        parser.add_argument(
            '--transform', help=('Specifies transform for preprocessing '
                                 '(arcsine, pseudocount, clr)'),
            required=False, type=str, default='pseudocount')
        parser.add_argument(
            '--no-grassmannian',
            help=('Specifies if grassmanian manifold optimization is disabled. '
                  'Turning this off remove unit norm constraint on decoder weights. '),
            required=False, dest='grassmanian', action='store_false')
        parser.set_defaults(grassmannian=True)
        parser.add_argument(
            '--distribution',
            help=('Specifies decoder distribution, either '
                  '`multinomial` or `gaussian`.'),
            required=False, type=str, default='multinomial')
        parser.add_argument(
            '--scheduler',
            help=('Learning rate scheduler '
                  '(choices include `cosine` and `steplr`'),
            default='cosine', required=False, type=str)
        return parser


# Batch correction methods
class MultBatchVAE(MultVAE):
    def __init__(self, n_input, batch_prior, n_batches,
                 n_latent=32, n_hidden=64, basis=None,
                 dropout=0.5, bias=True, batch_norm=False,
                 encoder_depth=1, learning_rate=0.001, scheduler='cosine',
                 distribution='multinomial', transform='pseudocount',
                 grassmannian=True):
        super().__init__(n_input, n_latent, n_hidden, basis=basis,
                         dropout=dropout, bias=bias, batch_norm=batch_norm,
                         encoder_depth=encoder_depth,
                         learning_rate=learning_rate, scheduler=scheduler,
                         transform=transform)
        self._hparams = {
            'n_input': n_input,
            'n_latent': n_latent,
            'n_hidden': n_hidden,
            'basis': basis,
            'dropout': dropout,
            'bias': bias,
            'batch_norm': batch_norm,
            'encoder_depth': encoder_depth,
            'n_batches': n_batches,
            'batch_prior': batch_prior,
            'learning_rate': learning_rate,
            'scheduler': scheduler,
            'distribution': distribution,
            'transform': transform,
            'grassmannian': grassmannian
        }
        self.gt_eigvectors = None
        self.gt_eigs = None

        batch_prior = pd.read_table(batch_prior, dtype=str)
        batch_prior = batch_prior.set_index(batch_prior.columns[0])
        batch_prior = batch_prior.values.astype(np.float64)
        batch_prior = batch_prior.reshape(1, -1).squeeze()
        batch_prior = torch.Tensor(batch_prior).float()
        basis = self.set_basis(n_input, basis)
        self.vae = LinearBatchVAE(
            n_input,
            hidden_dim=n_hidden,
            latent_dim=n_latent,
            batch_dim=n_batches,
            batch_norm=batch_norm,
            batch_prior=batch_prior,
            basis=basis,
            encoder_depth=encoder_depth,
            bias=bias,
            distribution=distribution,
            transform=transform,
            grassmannian=grassmannian)
        self.gt_eigvectors = None
        self.gt_eigs = None

    def initialize_batch(self, beta):
        # apparently this is not recommended, but fuck it
        self.vae.beta.weight.data = beta.data
        self.vae.beta.requires_grad = False
        self.vae.beta.weight.requires_grad = False

    def initialize_decoder(self, W):
        # can't initialize easily W due to geotorch
        # https://github.com/Lezcano/geotorch/issues/14
        self.vae.decoder.weight = W
        self.vae.decoder.weight.requires_grad = False

    def to_latent(self, X, b):
        """ Casts to latent space using predicted batch probabilities.

        Parameters
        ----------
        X : torch.Tensor
           Counts of interest (N x D)
        b : torch.Tensor
           Batch membership (N)
        """
        return self.vae.encode(X, b)

    def to_latent_marginalized(self, X, b):
        """ Casts to latent space using predicted batch probabilities.

        Parameters
        ----------
        X : torch.Tensor
           Counts of interest (N x D)
        b : torch.Tensor
           Class prediction probabilities for batch prediction (N x k)
        """
        return self.vae.encode_marginalized(X, b)

    def training_step(self, batch, batch_idx, optimizer_idx):
        if len(self.trainer.lr_schedulers) >= 1:
            lr = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
            current_lr = lr
        else:
            current_lr = self.hparams['learning_rate']

        counts, batch_ids = batch
        counts = counts.to(self.device)
        batch_ids = batch_ids.to(self.device)
        self.vae.train()
        losses = self.vae(counts, batch_ids)
        loss, recon_loss, kl_div_z, kl_div_b = losses
        assert torch.isnan(loss).item() is False
        tensorboard_logs = {
            'lr': current_lr, 'train_loss': loss
        }
        # log the learning rate
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        encode_params = self.vae.encoder.parameters()
        decode_params = self.vae.decoder.parameters()
        opt_g = torch.optim.Adam(
            list(encode_params) + list(decode_params),
            lr=self.hparams['learning_rate'])
        opt_b = torch.optim.Adam(
            list(self.vae.beta.parameters()) + [self.vae.batch_logvars],
            lr=self.hparams['learning_rate'])
        if self.hparams['scheduler'] == 'cosine_warm':
            scheduler = CosineAnnealingWarmRestarts(
                opt_g, T_0=2, T_mult=2)
        elif self.hparams['scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(
                opt_g, T_max=10)
        elif self.hparams['scheduler'] == 'none':
            return [opt_g, opt_b]
        else:
            raise ValueError(
                f'Scheduler {self.scheduler} not defined.')
        scheduler_b = CosineAnnealingLR(
            opt_b, T_max=10)
        return [opt_g, opt_b], [scheduler, scheduler_b]

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            counts, batch_ids = batch
            counts = counts.to(self.device)
            batch_ids = batch_ids.to(self.device)
            losses = self.vae(counts, batch_ids)
            loss, rec_err, kl_div_z, kl_div_b = losses
            assert torch.isnan(loss).item() is False
            # Record the actual loss.
            tensorboard_logs = {'val_loss': loss,
                                'val/recon_loss': rec_err,
                                'val/kl_div_z': kl_div_z,
                                'val/kl_div_b': kl_div_b,
                                'val_rec_err': rec_err}
            # log the learning rate
            return {'val_loss': loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        metrics = ['val_loss',
                   'val/recon_loss',
                   'val/kl_div_z',
                   'val/kl_div_b',
                   'val_rec_err']
        tensorboard_logs = {}
        for m in metrics:
            loss_f = lambda x: x['log'][m]
            losses = list(map(loss_f, outputs))
            rec_err = sum(losses) / len(losses)
            self.logger.experiment.add_scalar(
                m, rec_err, self.global_step)
            self.log(m, rec_err)
            tensorboard_logs[m] = rec_err

        if (self.gt_eigvectors is not None) and (self.gt_eigs is not None):
            ms = metric_subspace(self.vae, self.gt_eigvectors, self.gt_eigs)
            ma = metric_alignment(self.vae, self.gt_eigvectors)
            mp = metric_procrustes(self.vae, self.gt_eigvectors)
            mr = metric_pairwise(self.vae, self.gt_eigvectors, self.gt_eigs)
            tlog = {'subspace_distance': ms, 'alignment': ma, 'procrustes': mp}
            self.logger.experiment.add_scalar(
                'procrustes', mp, self.global_step)
            self.logger.experiment.add_scalar(
                'pairwise_r', mr, self.global_step)
            self.logger.experiment.add_scalar(
                'subspace_distance', ms, self.global_step)
            self.logger.experiment.add_scalar(
                'alignment', ma, self.global_step)
            tensorboard_logs = {**tensorboard_logs, **tlog}

        return {'val_loss': rec_err, 'log': tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser, add_help=True):
        parser = MultVAE.add_model_specific_args(
            parent_parser, add_help=add_help)
        parser.add_argument(
            '--batch-prior',
            help=('Pre-learned batch effect priors'
                  '(must have same number of dimensions as `train-biom`)'),
            required=True, type=str, default=None)
        return parser


class TripletVAE(pl.LightningModule):
    def __init__(self, vae_model, batch_model, n_input=32, n_hidden=64,
                 dropout=0.5, bias=True, batch_norm=False,
                 learning_rate=0.001,
                 scheduler='cosine'):
        super().__init__()
        self.vae = vae_model
        self.bcm = batch_model
        self.triplet_net = TripletNet(n_input, n_hidden)
        self._hparams = {
            'n_input': n_input,
            'n_hidden': n_hidden,
            'learning_rate': learning_rate,
            # TODO: we should add VAE learning rate at some point.
            'scheduler': scheduler
        }

    def training_step(self, batch, batch_idx):
        # self.vae.train() # let the weights be frozen
        self.triplet_net.train()

        if len(self.trainer.lr_schedulers) >= 1:
            lr = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
            current_lr = lr
        else:
            current_lr = self.hparams['learning_rate']

        i_counts, j_counts, k_counts, i_dict, j_dict, k_dict = batch
        i_batch = torch.Tensor(self.bcm(i_dict)).float().to(self.device)
        j_batch = torch.Tensor(self.bcm(j_dict)).float().to(self.device)
        k_batch = torch.Tensor(self.bcm(k_dict)).float().to(self.device)
        i_counts = i_counts.to(self.device)
        j_counts = j_counts.to(self.device)
        k_counts = k_counts.to(self.device)
        pos_u = self.vae.to_latent(i_counts, i_batch)
        pos_v = self.vae.to_latent(j_counts, j_batch)
        neg_v = self.vae.to_latent(k_counts, k_batch)
        # Triplet loss
        loss = self.triplet_net(pos_u, pos_v, neg_v)
        assert torch.isnan(loss).item() is False
        tensorboard_logs = {
            'lr': current_lr,
            'train_loss': loss
        }
        # log the learning rate
        return {'loss': loss, 'log': tensorboard_logs}

    def forward(self, x, b):
        x = self.vae.vae.encode_marginalized(x, b)
        return self.triplet_net.encode(x)

    def from_biom_to_latent(self, table):
        features = self.bcm.biom_to_features(table)
        b = self.bcm(features)
        X = torch.Tensor(table.matrix_data.todense().T).float()
        b = torch.Tensor(b).float()
        return self.to_latent(X, b)

    def to_latent(self, x, b):
        return self.forward(x, b)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            list(self.triplet_net.parameters()),
            lr=self.hparams['learning_rate'])
        if self.hparams['scheduler'] == 'cosine_warm':
            scheduler = CosineAnnealingWarmRestarts(
                opt, T_0=2, T_mult=2)
        elif self.hparams['scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(opt, T_max=10)
        elif self.hparams['scheduler'] == 'steplr':
            scheduler = StepLR(opt, step_size=10, gamma=0.5)
        elif self.hparams['scheduler'] == 'none':
            return [opt]
        else:
            raise ValueError(
                f'Scheduler {self.scheduler} not defined.')
        scheduler = CosineAnnealingLR(
            opt, T_max=10)
        return [opt], [scheduler]

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            i_counts, j_counts, k_counts, i_dict, j_dict, k_dict = batch
            i_batch = torch.Tensor(self.bcm(i_dict)).float().to(self.device)
            j_batch = torch.Tensor(self.bcm(j_dict)).float().to(self.device)
            k_batch = torch.Tensor(self.bcm(k_dict)).float().to(self.device)
            i_counts = i_counts.to(self.device)
            j_counts = j_counts.to(self.device)
            k_counts = k_counts.to(self.device)

            # losses = self.vae(counts, batch_ids)
            # vae_loss, recon_loss, kl_div_z, kl_div_b = losses
            pos_u = self.vae.to_latent(i_counts, i_batch)
            pos_v = self.vae.to_latent(j_counts, j_batch)
            neg_v = self.vae.to_latent(k_counts, k_batch)
            # Triplet loss
            loss = self.triplet_net(pos_u, pos_v, neg_v)
            assert torch.isnan(loss).item() is False

            if len(self.trainer.lr_schedulers) >= 1:
                lr = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
                current_lr = lr
            else:
                current_lr = self.hparams['learning_rate']

            tensorboard_logs = {
                'lr': current_lr,
                'val/loss': loss
            }

            # log the learning rate
            return {'val_loss': loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        metrics = ['val/loss']
        tensorboard_logs = {}
        for m in metrics:
            loss_f = lambda x: x['log'][m]
            losses = list(map(loss_f, outputs))
            rec_err = sum(losses) / len(losses)
            self.logger.experiment.add_scalar(
                m, rec_err, self.global_step)
            tensorboard_logs[m] = rec_err
        return {'val_loss': rec_err, 'log': tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser, add_help=True):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=add_help)
        parser.add_argument(
            '--n-hidden', help='Encoder dimension.',
            required=False, type=int, default=64)
        parser.add_argument(
            '--dropout', help='Dropout probability',
            required=False, type=float, default=0.1)
        parser.add_argument('--bias', dest='bias', action='store_true')
        parser.add_argument('--no-bias', dest='bias', action='store_false')
        parser.add_argument('--batch-norm', dest='batch_norm',
                            action='store_true')
        parser.add_argument('--no-batch-norm', dest='batch_norm',
                            action='store_false')
        parser.add_argument(
            '--learning-rate', help='Learning rate',
            required=False, type=float, default=1e-3)
        parser.add_argument(
            '--scheduler',
            help=('Learning rate scheduler '
                  '(choices include `cosine` and `steplr`'),
            default='cosine', required=False, type=str)
        return parser


def add_data_specific_args(parent_parser, add_help=True):
    parser = argparse.ArgumentParser(parents=[parent_parser],
                                     add_help=add_help)
    # Arguments specific for dataloaders
    parser.add_argument(
        '--train-biom', help='Training biom file', required=True)
    parser.add_argument(
        '--test-biom', help='Testing biom file', required=True)
    parser.add_argument(
        '--val-biom', help='Validation biom file', required=True)
    parser.add_argument(
        '--sample-metadata', help='Sample metadata file', required=False)
    parser.add_argument(
        '--batch-category',
        help='Sample metadata column for batch effects.',
        required=False, type=str, default=None)
    parser.add_argument(
        '--class-category',
        help='Sample metadata column for class predictions.',
        required=False, type=str, default=None)
    parser.add_argument(
        '--batch-size', help='Training batch size',
        required=False, type=int, default=32)
    # Arguments specific for trainer
    parser.add_argument(
        '--epochs', help='Training batch size',
        required=False, type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--profile', type=bool, default=False)
    parser.add_argument('--grad-clip', type=int, default=10)
    parser.add_argument('--eigvalues', type=str, default=None,
                        help='Ground truth eigenvalues (optional)',
                        required=False)
    parser.add_argument('--eigvectors', type=str, default=None,
                        help='Ground truth eigenvectors (optional)',
                        required=False)
    parser.add_argument('--load-from-checkpoint', type=str, default=None)
    parser.add_argument('--output-directory', type=str, default=None)
    return parser
