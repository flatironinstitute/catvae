import datetime
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, StepLR
)
from catvae.dataset.biom import collate_single_f, BiomDataset
from catvae.models import LinearCatVAE, LinearVAE
from catvae.composition import ilr_inv
from catvae.metrics import (
    metric_subspace, metric_transpose_theorem,
    metric_alignment, metric_orthogonality)
import pytorch_lightning as pl
from skbio import TreeNode
from skbio.stats.composition import alr_inv, closure
from gneiss.balances import sparse_balance_basis

from biom import load_table
from scipy.stats import entropy
import numpy as np


class LightningCountVAE(pl.LightningModule):

    def __init__(self, args):
        super(LightningCountVAE, self).__init__()
        self.hparams = args
        if self.hparams.basis_file is not None:
            tree = TreeNode.read(self.hparams.basis_file)
            basis, nodes = sparse_balance_basis(tree)
        else:
            basis = None

        # a sneak peek into file types to initialize model
        n_input = load_table(self.hparams.train_biom).shape[0]

        self.model = LinearCatVAE(
            n_input,
            hidden_dim=self.hparams.n_latent,
            basis=basis,
            imputer=self.hparams.imputer,
            batch_size=self.hparams.batch_size)

        self.gt_eigvectors = None
        self.gt_eigs = None

    def set_eigs(self, gt_eigvectors, gt_eigs):
        self.gt_eigvectors = gt_eigvectors
        self.gt_eigs = gt_eigs

    def forward(self, X):
        return self.model(X)

    def initialize_logging(self, root_dir='./', logging_path=None):
        if logging_path is None:
            basename = "logdir"
            suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            logging_path = "_".join([basename, suffix])
        full_path = root_dir + logging_path
        writer = SummaryWriter(full_path)
        return writer

    def train_dataloader(self):
        train_dataset = BiomDataset(load_table(self.hparams.train_biom))
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.hparams.batch_size,
            collate_fn=collate_single_f, shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = BiomDataset(load_table(self.hparams.val_biom))
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.hparams.batch_size,
            collate_fn=collate_single_f, shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True)
        return val_dataloader

    def test_dataloader(self):
        test_dataset = BiomDataset(load_table(self.hparams.test_biom))
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.hparams.batch_size,
            collate_fn=collate_single_f,
            shuffle=False, num_workers=self.hparams.num_workers,
            pin_memory=True)
        return test_dataloader

    def training_step(self, batch, batch_idx):
        self.model.train()
        counts = batch
        loss = self.model(counts)
        assert torch.isnan(loss).item() is False
        if len(self.trainer.lr_schedulers) >= 1:
            lr = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
            current_lr = lr
        else:
            current_lr = self.hparams.learning_rate
        tensorboard_logs = {
            'train_loss': loss, 'elbo': -loss, 'lr': current_lr
        }
        # log the learning rate
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            counts = batch
            loss = self.model(counts)
            assert torch.isnan(loss).item() is False

            # Record the actual loss.
            rec_err = self.model.get_reconstruction_loss(batch)
            tensorboard_logs = {'validation_loss': loss,
                                'val_rec_err': rec_err}

            # log the learning rate
            return {'validation_loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        loss_f = lambda x: x['validation_loss']
        losses = list(map(loss_f, outputs))
        loss = sum(losses) / len(losses)
        self.logger.experiment.add_scalar('val_loss', loss, self.global_step)
        loss_f = lambda x: x['log']['val_rec_err']
        losses = list(map(loss_f, outputs))
        rec_err = sum(losses) / len(losses)
        self.logger.experiment.add_scalar('val_rec_err',
                                          rec_err, self.global_step)
        mt = metric_transpose_theorem(self.model)
        self.logger.experiment.add_scalar('transpose', mt, self.global_step)
        ortho, eig_err = metric_orthogonality(self.model)
        self.logger.experiment.add_scalar('orthogonality',
                                          ortho, self.global_step)

        tensorboard_logs = dict(
            [('val_loss', loss),
             ('val_rec_err', rec_err),
             ('transpose', mt),
             ('orthogonality', ortho),
             ('eigenvalue-error', eig_err)]
        )

        if (self.gt_eigvectors is not None) and (self.gt_eigs is not None):
            ms = metric_subspace(
                self.model, self.gt_eigvectors, self.gt_eigs)
            ma = metric_alignment(self.model, self.gt_eigvectors)
            tlog = {'subspace_distance': ms, 'alignment': ma}
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
            self.model.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.scheduler == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=2, T_mult=2)
        elif self.hparams.scheduler == 'steplr':
            m = 1e-6  # minimum learning rate
            steps = int(np.log2(self.hparams.learning_rate / m))
            steps = self.hparams.epochs // steps
            scheduler = StepLR(optimizer, step_size=steps, gamma=0.5)
        elif self.hparams.scheduler == 'none':
            return [optimizer]
        else:
            s = self.hparams.scheduler
            raise ValueError(f'{s} is not implemented.')
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser, add_help=True):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=add_help)
        parser.add_argument(
            '--train-biom', help='Training biom file', required=True)
        parser.add_argument(
            '--test-biom', help='Testing biom file', required=True)
        parser.add_argument(
            '--val-biom', help='Validation biom file', required=True)
        parser.add_argument(
            '--basis-file',
            help=('Newick file to specify basis from bifurcating tree.'
                  'If not specified, a random basis will be defined.'),
            required=False, type=str, default=None)
        parser.add_argument(
            '--n-latent', help='Latent embedding dimension.',
            required=False, type=int, default=10)
        parser.add_argument(
            '--n-layers', help='Number of encoding layers.',
            required=False, type=int, default=1)
        parser.add_argument(
            '--n-samples',
            help='Number of monte carlo samples for computing expectations.',
            required=False, type=int, default=1)
        parser.add_argument(
            '--learning-rate', help='Learning rate',
            required=False, type=float, default=1e-3)
        parser.add_argument(
            '--batch-size', help='Training batch size',
            required=False, type=int, default=32)
        parser.add_argument(
            '--use-analytic-elbo', help='Use analytic formulation of elbo.',
            required=False, type=bool, default=True)
        parser.add_argument(
            '--imputer', help='Imputation technique to use.',
            required=False, type=bool, default=None)
        parser.add_argument(
            '--scheduler',
            help=('Learning rate scheduler '
                  '(choices include `cosine` and `steplr`'),
            default='cosine', required=False, type=str)
        parser.add_argument(
            '--epochs', help='Training batch size',
            required=False, type=int, default=10)
        parser.add_argument(
            '-o', '--output-directory',
            help='Output directory of model results', required=True)
        return parser
