import datetime
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, StepLR
)
from catvae.dataset.biom import (
    collate_single_f, BiomDataset,
    collate_batch_f
)
from catvae.models import LinearCatVAE, LinearVAE
from catvae.composition import (ilr_inv, alr_basis,
                                ilr_basis, identity_basis)
from catvae.metrics import (
    metric_subspace, metric_transpose_theorem, metric_pairwise,
    metric_procrustes, metric_alignment, metric_orthogonality)
import pytorch_lightning as pl
from skbio import TreeNode
from skbio.stats.composition import alr_inv, closure
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score)
from sklearn.exceptions import NotFittedError
from biom import load_table
import pandas as pd
from scipy.stats import entropy
from scipy.sparse import coo_matrix
import numpy as np
import os


def to_numpy(x):
    return x.detach().cpu().numpy()


class LightningVAE(pl.LightningModule):

    def __init__(self, args):
        super(LightningVAE, self).__init__()
        self.hparams = args
        self.gt_eigvectors = None
        self.gt_eigs = None

    def set_basis(self, n_input, table):
        # a sneak peek into file types to initialize model
        if (self.hparams.basis is not None and
            os.path.exists(self.hparams.basis)):
            basis = ilr_basis(self.hparams.basis, table)
        elif self.hparams.basis == 'alr':
            basis = coo_matrix(alr_basis(n_input))
        elif self.hparams.basis == 'identity':
            basis = coo_matrix(identity_basis(n_input))
        else:
            basis = None
        return basis

    def set_eigs(self, gt_eigvectors, gt_eigs):
        self.gt_eigvectors = gt_eigvectors
        self.gt_eigs = gt_eigs

    def forward(self, X):
        return self.model(X)

    def to_latent(self, X):
        return self.model.encode(X)

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
            collate_fn=collate_single_f, shuffle=True,
            num_workers=self.hparams.num_workers, drop_last=True,
            pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = BiomDataset(load_table(self.hparams.val_biom))
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.hparams.batch_size,
            collate_fn=collate_single_f, shuffle=False,
            num_workers=self.hparams.num_workers, drop_last=True,
            pin_memory=True)
        return val_dataloader

    def test_dataloader(self):
        test_dataset = BiomDataset(load_table(self.hparams.test_biom))
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.hparams.batch_size,
            collate_fn=collate_single_f,
            shuffle=False, num_workers=self.hparams.num_workers,
            drop_last=True, pin_memory=True)
        return test_dataloader

    def training_step(self, batch, batch_idx):
        self.model.train()
        counts = batch.to(self.device)
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
        loss_f = lambda x: x['log']['val_rec_err']
        losses = list(map(loss_f, outputs))
        rec_err = sum(losses) / len(losses)
        self.logger.experiment.add_scalar('val_rec_err',
                                          rec_err, self.global_step)
        if self.hparams.encoder_depth == 1:
            mt = metric_transpose_theorem(self.model)
            self.logger.experiment.add_scalar('transpose', mt, self.global_step)
        ortho, eig_err = metric_orthogonality(self.model)
        self.logger.experiment.add_scalar('orthogonality',
                                          ortho, self.global_step)

        tensorboard_logs = dict(
            [('val_loss', rec_err),
             # ('transpose', mt),
             ('orthogonality', ortho),
             ('eigenvalue-error', eig_err)]
        )

        if (self.gt_eigvectors is not None) and (self.gt_eigs is not None):
            ms = metric_subspace(self.model, self.gt_eigvectors, self.gt_eigs)
            ma = metric_alignment(self.model, self.gt_eigvectors)
            mp = metric_procrustes(self.model, self.gt_eigvectors)
            mr = metric_pairwise(self.model, self.gt_eigvectors, self.gt_eigs)
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

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.scheduler == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=2, T_mult=2)
        # elif self.hparams.scheduler == 'cosine':
        #     scheduler = CosineAnnealingLR(
        #         optimizer, T_max=self.hparams.steps_per_batch * 10)
        elif self.hparams.scheduler == 'steplr':
            m = 1e-1  # maximum learning rate
            steps = int(np.log2(m / self.hparams.learning_rate))
            steps = self.hparams.epochs // steps
            scheduler = StepLR(optimizer, step_size=steps, gamma=0.5)
        elif self.hparams.scheduler == 'inv_steplr':
            m = 1e-1  # maximum learning rate
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=m)
            steps = int(np.log2(m / self.hparams.learning_rate))
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
            '--basis',
            help=('Basis.  Options include `alr`, `identity` or a file. '
                  'If a file is specified, it is assumed to be in '
                  'Newick file to specify basis from bifurcating tree. '
                  'If not specified, a random basis will be defined.'),
            required=False, type=str, default=None)
        parser.add_argument(
            '--n-latent', help='Latent embedding dimension.',
            required=False, type=int, default=10)
        parser.add_argument('--bias', dest='bias', action='store_true')
        parser.add_argument('--no-bias', dest='bias', action='store_false')
        parser.add_argument(
            '--encoder-depth', help='Number of encoding layers.',
            required=False, type=int, default=1)
        parser.add_argument(
            '--learning-rate', help='Learning rate',
            required=False, type=float, default=1e-3)
        parser.add_argument(
            '--batch-size', help='Training batch size',
            required=False, type=int, default=32)
        parser.add_argument(
            '--steps-per-batch',
            help='Number of gradient descent steps per batch.',
            required=False, type=int, default=10)
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


# Main VAE classes
class LightningCatVAE(LightningVAE):
    def __init__(self, args):
        super(LightningCatVAE, self).__init__(args)
        self.hparams = args
        table = load_table(self.hparams.train_biom)
        n_input = table.shape[0]
        basis = self.set_basis(n_input, table)
        imputer = lambda x: x + 1
        self.model = LinearCatVAE(
            n_input,
            hidden_dim=self.hparams.n_latent,
            basis=basis,
            #imputer=self.hparams.imputer,
            imputer=None,
            encoder_depth=self.hparams.encoder_depth,
            batch_size=self.hparams.batch_size,
            bias=self.hparams.bias
        )
        self.gt_eigvectors = None
        self.gt_eigs = None

    def configure_optimizers(self):
        # optimizer_eta = torch.optim.LBFGS(
        #     [self.model.eta], lr=self.hparams.learning_rate,
        #     tolerance_grad=1e-7,
        #     history_size=100, max_iter=1000)
        optimizer_eta = torch.optim.Adam(
            [self.model.eta], lr=self.hparams.learning_rate)

        optimizer = torch.optim.Adam(
            list(self.model.encoder.parameters()) +
            list(self.model.decoder.parameters()) +
            [self.model.log_sigma_sq, self.model.variational_logvars],
            lr=self.hparams.learning_rate)
        if self.hparams.scheduler == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=2, T_mult=2)
        elif self.hparams.scheduler == 'steplr':
            m = 1e-5  # min learning rate
            steps = int(np.log2(m / self.hparams.learning_rate))
            steps = 100 * self.hparams.epochs // steps
            scheduler = StepLR(optimizer, step_size=steps, gamma=0.5)
        elif self.hparams.scheduler == 'none':
            return [optimizer_eta, optimizer]
        else:
            s = self.hparams.scheduler
            raise ValueError(f'{s} is not implemented.')
        return [optimizer_eta, optimizer], [scheduler]

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i,
                       second_order_closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        # perform multiple steps with LBFGS to optimize eta
        if optimizer_i == 0:
            for _ in range(self.hparams.steps_per_batch):
                optimizer.step(second_order_closure)
                optimizer.zero_grad()

        # update all of the other parameters once
        # eta is optimized
        if optimizer_i == 1:
            for _ in range(self.hparams.steps_per_batch):
                # print('current_epoch', current_epoch,
                #       'batch', batch_nb, 'optimizer', optimizer_i, loss)
                optimizer.step(second_order_closure)
                optimizer.zero_grad()

        loss_ = second_order_closure().item()
        self.logger.experiment.add_scalar(
            'train_loss', loss_, self.global_step)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        counts = batch.to(self.device)
        self.model.reset(counts)

    def training_step(self, batch, batch_idx, optimizer_idx):
        return super().training_step(batch, batch_idx)


class LightningLinearVAE(LightningVAE):
    def __init__(self, args):
        super(LightningLinearVAE, self).__init__(args)
        self.hparams = args

        # a sneak peek into file types to initialize model
        table = load_table(self.hparams.train_biom)
        n_input = table.shape[0]
        basis = self.set_basis(n_input, table)
        self.model = LinearVAE(
            n_input, basis=basis,
            hidden_dim=self.hparams.n_latent,
            bias=self.hparams.bias)
        self.gt_eigvectors = None
        self.gt_eigs = None


# Batch correction methods
class LightningBatchVAE(LightningVAE):
    def __init__(self, args):
        super(LightningBatchVAE, self).__init__(args)
        self.hparams = args
        self.gt_eigvectors = None
        self.gt_eigs = None
        # an ensemble of lasso models
        self.regularizers = [1, 0.1, 0.01, 0.001]
        self.lassos = [
            SGDClassifier(
                loss='log', alpha=c, penalty='l1')
            for c in self.regularizers
        ]
        # index for best batch effect classifier
        # initialize to None if batch models aren't fitted
        self.best_batch_idx = None
        # get number of batches
        self.metadata = pd.read_table(
            self.hparams.sample_metadata, dtype=str)
        self.classes = np.arange(len(
            np.unique(self.metadata[args.batch_category])))

    def _batch_indices(self, eps=0.1):
        """ Obtain best batch predictor """
        batch_clf = self.lassos[self.best_batch_idx]
        idx = set()
        nums = np.arange(batch_clf.coef_.shape[1])
        for i in range(batch_clf.coef_.shape[0]):
            arr = batch_clf.coef_[i]
            j = abs(arr) > eps
            idx = idx | set(nums[j])
        idx = np.array(list(idx))
        return idx

    def to_latent(self, X, exclude_batch=True, eps=0.1):
        """ Obtain latent representation """
        if not exclude_batch:
            return self.model.encode(X)
        idx = self._batch_indices(eps)
        z = self.model.encode(X)
        # exclude the dimensions that best predict batches
        cidx = torch.Tensor(
            list(set(np.arange(z.shape[1])) - set(idx))
        ).long()
        return z[:, cidx]

    def get_embedding(self, exclude_batch=True, eps=0.1):
        """ Obtain microbial embedding matrix """
        W = self.model.decoder.weight
        if not exclude_batch:
            return W
        idx = self._batch_indices(eps)
        # exclude the dimensions that best predict batches
        cidx = torch.Tensor(
            list(set(np.arange(W.shape[1])) - set(idx))
        ).long()
        return W[:, cidx]

    def _dataloader(self, biom_file, shuffle=True):
        table = load_table(biom_file)
        self.metadata = pd.read_table(
            self.hparams.sample_metadata, dtype=str)
        index_name = self.metadata.columns[0]
        metadata = self.metadata.set_index(index_name)
        _dataset = BiomDataset(
            table, metadata,
            batch_category=self.hparams.batch_category)
        _dataloader = DataLoader(
            _dataset, batch_size=self.hparams.batch_size,
            collate_fn=collate_batch_f, shuffle=shuffle,
            num_workers=self.hparams.num_workers, drop_last=True,
            pin_memory=True)
        return _dataloader

    def train_dataloader(self):
        return self._dataloader(self.hparams.train_biom)

    def val_dataloader(self):
        return self._dataloader(self.hparams.val_biom, shuffle=False)

    def test_dataloader(self):
        return self._dataloader(self.hparams.test_biom, shuffle=False)

    def training_step(self, batch, batch_idx):
        self.model.train()
        counts, batch_ids = batch

        # Train sklearn batch classifiers
        for i in range(len(self.lassos)):
            z = self.model.encode(counts)
            self.lassos[i].partial_fit(
                X=to_numpy(z), y=to_numpy(batch_ids),
                classes=self.classes)

        counts = counts.to(self.device)
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
            counts, batch_ids = batch
            batch_ids = to_numpy(batch_ids)
            ccounts = to_numpy(counts)
            batch_logs = {}
            # obtain batch effect prediction results
            z = to_numpy(self.model.encode(counts))
            for i in range(len(self.lassos)):
                try:
                    y_pred = self.lassos[i].predict(z)
                except NotFittedError:
                    self.lassos[i].partial_fit(
                        X=z, y=batch_ids,
                        classes=self.classes)
                    y_pred = self.lassos[i].predict(z)
                label = self.regularizers[i]
                output = {
                    f'batch_accuracy_{label}': accuracy_score(
                        batch_ids, y_pred),
                    f'batch_F1_{label}': f1_score(
                        batch_ids, y_pred, average='micro'),
                    f'batch_precision_{label}': precision_score(
                        batch_ids, y_pred, average='micro'),
                    f'batch_recall_{label}': recall_score(
                        batch_ids, y_pred, average='micro')
                }
                batch_logs = {**batch_logs, **output}
            counts = counts.to(self.device)
            loss = self.model(counts)
            assert torch.isnan(loss).item() is False
            # Record the actual loss.
            rec_err = self.model.get_reconstruction_loss(counts)
            tensorboard_logs = {'validation_loss': loss,
                                'val_rec_err': rec_err}
            tensorboard_logs = {**tensorboard_logs, **batch_logs}
            # log the learning rate
            return {'validation_loss': loss, 'log': tensorboard_logs}


    def validation_epoch_end(self, outputs):
        loss_f = lambda x: x['log']['val_rec_err']
        losses = list(map(loss_f, outputs))
        rec_err = sum(losses) / len(losses)
        self.logger.experiment.add_scalar('val_rec_err',
                                          rec_err, self.global_step)
        # Get batch effect results, and decide optimal model based on F1 score
        metrics = ['batch_accuracy', 'batch_F1',
                   'batch_precision', 'batch_recall']
        results = {}
        for i in range(len(self.lassos)):
            label = self.regularizers[i]
            res = []
            for m in metrics:
                name = f'{m}_{label}'
                metric_f = lambda x: x['log'][name]
                mets = list(map(metric_f, outputs))
                avg_met = sum(mets) / len(mets)
                self.logger.experiment.add_scalar(
                    name, avg_met, self.global_step)
                res.append(avg_met)
            res = dict(zip(metrics, res))
            results[label] = res
        # get optimal model
        self.best_batch_idx = np.argmax([results[i]['batch_F1']
                                         for i in self.regularizers])
        # get simulation results
        if self.hparams.encoder_depth == 1:
            mt = metric_transpose_theorem(self.model)
            self.logger.experiment.add_scalar('transpose', mt, self.global_step)
        ortho, eig_err = metric_orthogonality(self.model)
        self.logger.experiment.add_scalar('orthogonality',
                                          ortho, self.global_step)

        tensorboard_logs = dict(
            [('val_loss', rec_err),
             # ('transpose', mt),
             ('orthogonality', ortho),
             ('eigenvalue-error', eig_err)]
        )

        if (self.gt_eigvectors is not None) and (self.gt_eigs is not None):
            ms = metric_subspace(self.model, self.gt_eigvectors, self.gt_eigs)
            ma = metric_alignment(self.model, self.gt_eigvectors)
            mp = metric_procrustes(self.model, self.gt_eigvectors)
            mr = metric_pairwise(self.model, self.gt_eigvectors, self.gt_eigs)
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
        parser = LightningVAE.add_model_specific_args(parent_parser)
        parser.add_argument(
            '--sample-metadata', help='Sample metadata file', required=False)
        parser.add_argument(
            '--batch-category',
            help='Sample metadata column for batch effects.',
            required=False, type=str, default=None)
        parser.add_argument(
            '--batch-priors',
            help=('Pre-learned batch effect priors'
                  '(must have same number of dimensions as `train-biom`)'),
            required=False, type=str, default=None)
        return parser


class LightningBatchCatVAE(LightningBatchVAE, LightningCatVAE):
    def __init__(self, args):
        LightningBatchVAE.__init__(self, args)
        LightningCatVAE.__init__(self, args)
        self.hparams = args
        table = load_table(self.hparams.train_biom)
        n_input = table.shape[0]
        basis = self.set_basis(n_input, table)
        self.model = LinearCatVAE(
            n_input,
            hidden_dim=self.hparams.n_latent,
            basis=basis,
            imputer=self.hparams.imputer,
            encoder_depth=self.hparams.encoder_depth,
            batch_size=self.hparams.batch_size,
            bias=self.hparams.bias
        )
        self.gt_eigvectors = None
        self.gt_eigs = None

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        counts = batch[0]
        counts = counts.to(self.device)
        self.model.reset(counts)

    def training_step(self, batch, batch_idx, optimizer_idx):
        return super().training_step(batch, batch_idx)


class LightningBatchLinearVAE(LightningBatchVAE, LightningLinearVAE):
    def __init__(self, args):
        LightningBatchVAE.__init__(self, args)
        LightningLinearVAE.__init__(self, args)
        self.hparams = args
        table = load_table(self.hparams.train_biom)
        n_input = table.shape[0]
        basis = self.set_basis(n_input, table)
        self.model = LinearVAE(
            n_input,
            hidden_dim=self.hparams.n_latent,
            basis=basis,
            encoder_depth=self.hparams.encoder_depth,
            bias=self.hparams.bias
        )
        self.gt_eigvectors = None
        self.gt_eigs = None
