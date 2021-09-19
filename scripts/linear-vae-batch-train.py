import os
import argparse
import numpy as np
import pandas as pd
from catvae.trainer import (MultVAE, MultBatchVAE, BiomDataModule,
                            add_data_specific_args)
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning import loggers as pl_loggers
from biom import load_table
from skbio import TreeNode
import yaml


def main(args):
    if args.load_from_checkpoint is not None:
        model = MultBatchVAE.load_from_checkpoint(args.load_from_checkpoint)
    else:
        table = load_table(args.train_biom)
        n_input = table.shape[0]
        sample_metadata = pd.read_table(args.sample_metadata, dtype=str)
        sample_metadata = sample_metadata.set_index(sample_metadata.columns[0])
        sample_metadata = sample_metadata.loc[table.ids()]
        n_batches = len(sample_metadata[args.batch_category].value_counts())
        gam, phi = args.gam_prior.split(',')
        model = MultBatchVAE(
            n_input=n_input,
            n_batches=n_batches,
            n_latent=args.n_latent,
            n_hidden=args.n_hidden,
            beta_prior=args.beta_prior,
            gam_prior=float(gam),
            phi_prior=float(phi),
            basis=args.basis,
            dropout=args.dropout,
            bias=args.bias,
            batch_norm=args.batch_norm,
            encoder_depth=args.encoder_depth,
            learning_rate=args.learning_rate,
            vae_lr=args.vae_lr,
            scheduler=args.scheduler,
            transform=args.transform,
            grassmannian=args.grassmannian)
        if args.load_vae_weights is not None:
            # initialize encoder/decoder weights with pretrained VAE
            other_model = MultVAE.load_from_checkpoint(args.load_vae_weights)
            model.vae.encoder = other_model.vae.encoder
            model.vae.decoder = other_model.vae.decoder
            model.vae.mu_net = other_model.vae.mu_net
            model.vae.sigma_net = other_model.vae.sigma_net
            model.vae.log_sigma_sq = other_model.vae.log_sigma_sq
            # Note that input_embed isn't handled here.

    print(args)
    print(model)
    if args.eigvectors is not None and args.eigvalues is not None:
        eigvectors = np.loadtxt(args.eigvectors)
        eigvalues = np.loadtxt(args.eigvalues)
        model.set_eigs(eigvectors, eigvalues)
    if args.profile:
        profiler = AdvancedProfiler()
    else:
        profiler = None
    dm = BiomDataModule(
        args.train_biom, args.test_biom, args.val_biom,
        metadata=args.sample_metadata, batch_category=args.batch_category,
        batch_size=args.batch_size, num_workers=args.num_workers)

    ckpt_path = os.path.join(
        args.output_directory,
        "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        period=1,
        monitor='val_loss',
        mode='min',
        verbose=True)

    os.mkdir(args.output_directory)
    tb_logger = pl_loggers.TensorBoardLogger(f'{args.output_directory}/logs/')
    # save hyper-parameters to yaml file
    with open(f'{args.output_directory}/hparams.yaml', 'w') as outfile:
        yaml.dump(model._hparams, outfile, default_flow_style=False)
    # save batch class mappings
    dm.batch_categories.to_csv(
        f'{args.output_directory}/batch_categories.txt', sep='\t', header=None)
    # save tree to file if specified
    if os.path.exists(args.basis):
        tree = TreeNode.read(args.basis)
        tree.write(f'{args.output_directory}/tree.nwk')

    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        check_val_every_n_epoch=1,
        gradient_clip_val=args.grad_clip,
        profiler=profiler,
        logger=tb_logger,
        callbacks=[checkpoint_callback])

    trainer.fit(model, dm)
    trainer.save_checkpoint(
        args.output_directory + '/last_ckpt.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser = MultBatchVAE.add_model_specific_args(parser, add_help=False)
    parser = add_data_specific_args(parser, add_help=False)
    args = parser.parse_args()
    main(args)
