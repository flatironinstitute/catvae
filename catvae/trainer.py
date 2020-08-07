import datetime
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, StepLR, CyclicLR
)
from mavi.dataset.biom import collate_single_f, BiomDataset
from mavi.vae import CountVAE
from mavi.linear_vae import LinearVAE
from mavi.utils import alr_basis
from mavi.metrics import (
    metric_subspace, metric_transpose_theorem, metric_alignment)
import pytorch_lightning as pl
from skbio import TreeNode
from skbio.stats.composition import alr_inv, closure
from gneiss.balances import _balance_basis

from biom import load_table
from scipy.stats import entropy
import numpy as np


class LightningCountVAE(pl.LightningModule):

    def __init__(self, args):
        super(LightningCountVAE, self).__init__()
        self.hparams = args
        if self.hparams.basis_file is not None:
            tree = TreeNode.read(self.hparams.basis_file)
            basis, nodes = _balance_basis(tree)
        else:
            basis = None

        # a sneak peek into file types to initialize model
        n_input = load_table(self.hparams.train_biom).shape[0]
        n_batch = 0
        if self.hparams.sample_metadata is not None:
            n_batch = len(pd.read_table(self.hparams.sample_metadata)[
                self.hparams.batch_category
            ].value_counts())

        self.model = CountVAE(n_input,
                              n_batch,
                              self.hparams.n_latent,
                              self.hparams.n_layers,
                              self.hparams.dropout_rate,
                              self.hparams.dispersion,
                              self.hparams.reconstruction_loss,
                              basis)
        self.gt_eigvectors = None
        self.gt_eigs = None

    def set_eigs(self, gt_eigvectors, gt_eigs):
        self.gt_eigvectors = gt_eigvectors
        self.gt_eigs = gt_eigs

    def forward(self, X, X_smoothed):
        return self.model(X, X_smoothed)

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
            train_dataset, self.hparams.batch_size,
            collate_fn=collate_single_f, shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = BiomDataset(load_table(self.hparams.val_biom))
        val_dataloader = DataLoader(
            val_dataset, self.hparams.batch_size,
            collate_fn=collate_single_f, shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True)
        return val_dataloader

    def test_dataloader(self):
        test_dataset = BiomDataset(load_table(self.hparams.test_biom))
        test_dataloader = DataLoader(
            test_dataset, self.hparams.batch_size, collate_fn=collate_single_f,
            shuffle=False, num_workers=self.hparams.num_workers,
            pin_memory=True)
        return test_dataloader

    def compute_loss(self, reconst_loss, kl_divergence):
        loss = torch.mean(reconst_loss + kl_divergence)
        return loss

    def training_step(self, batch, batch_idx):
        self.model.train()
        counts, batches, smoothed_counts = batch
        rec_loss, kl_local = self.model(
            counts, smoothed_counts, n_samples=self.hparams.n_samples)
        loss = self.compute_loss(rec_loss, kl_local)
        assert torch.isnan(loss).item() is False
        if len(self.trainer.lr_schedulers) >= 1:
            current_lr = self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()[0]
        else:
            current_lr = self.hparams.learning_rate

        tensorboard_logs = {
            'train_loss': loss, 'elbo': -loss, 'lr': current_lr,
            'train_reconstruction_loss': rec_loss.mean(),
            'train_latent_loss': kl_local.mean()
        }
        # log the learning rate
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        self.model.train()
        counts, batches, smoothed_counts = batch
        rec_loss,  kl_local = self.model(counts, smoothed_counts)
        loss = self.compute_loss(rec_loss, kl_local)
        assert torch.isnan(loss).item() is False

        # Record the actual loss.
        res = self.model.inference(smoothed_counts)
        pred_probs = closure(alr_inv(res['px_mean'].cpu().detach().numpy()))
        kl_diffs = []
        cnts = closure(counts.cpu().detach().numpy())
        for i in range(counts.shape[0]):
            e = entropy(cnts[i], pred_probs[i])
            kl_diffs.append(e)
        kl_diff = np.mean(kl_diffs)
        tensorboard_logs = {'validation_loss': loss, 'pred_kl': kl_diff,
            'val_reconstruction_loss': rec_loss.mean(),
            'val_latent_loss': kl_local.mean()
        }

        # log the learning rate
        return {'validation_loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        loss_f = lambda x: x['validation_loss']
        losses = list(map(loss_f, outputs))
        loss = sum(losses) / len(losses)
        self.logger.experiment.add_scalar('val_loss', loss, self.global_step)
        loss_f = lambda x: x['log']['pred_kl']
        losses = list(map(loss_f, outputs))
        kl_diff = sum(losses) / len(losses)
        self.logger.experiment.add_scalar('pred_kl', kl_diff, self.global_step)
        mt = metric_transpose_theorem(self.model)
        self.logger.experiment.add_scalar('transpose', mt, self.global_step)
        tensorboard_logs = dict(
            [('val_loss', loss), ('pred_kl', kl_diff), ('transpose', mt)]
        )

        if (self.gt_eigvectors is not None) and (self.gt_eigs is not None):
            ms = metric_subspace(
                self.model, self.gt_eigvectors, self.gt_eigs)
            ma = metric_alignment(self.model, self.gt_eigvectors)
            tlog = {'subspace_distance': ms, 'alignment' : ma}
            self.logger.experiment.add_scalar('subspace_distance', ms, self.global_step)
            self.logger.experiment.add_scalar('alignment', ma, self.global_step)
            tensorboard_logs = {**tensorboard_logs, **tlog}

        return {'val_loss': loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.scheduler == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=5, T_mult=2)
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
            '--sample-metadata', help='Sample metadata file', required=False)
        parser.add_argument(
            '--batch-category', help='Batch category', required=False)
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
            '--dropout-rate', help='Encoder dropout rate.',
            required=False, type=float, default=0)
        parser.add_argument(
            '--use-relu', help='Use relu or linear activation for Encoder',
            required=False, type=bool, default=False)
        parser.add_argument(
            '--dispersion',
            help='Dispersion specification (options include gene, gene-batch)',
            required=False, type=str, default='gene')
        parser.add_argument(
            '--reconstruction-loss',
            help='Reconstruction loss (options include nb, mln, multinomial).',
            required=False, type=str, default='nb')
        parser.add_argument(
            '--basis-file',
            help=('Newick file to specify basis from bifurcating tree.'
                  'If not specified, an ALR basis will be used.'),
            required=False, type=str, default=None)
        parser.add_argument(
            '--learning-rate', help='Learning rate',
            required=False, type=float, default=1e-3)
        parser.add_argument(
            '--batch-size', help='Training batch size',
            required=False, type=int, default=32)
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
