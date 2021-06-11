import os
import argparse
import numpy as np
import pandas as pd
from catvae.trainer import (MultBatchVAE, TripletVAE, TripletDataModule,
                            add_data_specific_args)
import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning import loggers as pl_loggers
from biom import load_table
from skbio import TreeNode
# this requires qiime2
import qiime2
from sklearn.pipeline import Pipeline
import yaml


def main(args):
    vae_model = MultBatchVAE.load_from_checkpoint(args.vae_model_path)
    batch_model = qiime2.Artifact.load(args.batch_model_path).view(Pipeline)
    dm = TripletDataModule(
        args.train_biom, args.test_biom, args.val_biom,
        metadata=args.sample_metadata,
        batch_category=args.batch_category,
        class_category=args.class_category,
        batch_size=args.batch_size, num_workers=args.num_workers)

    table = load_table(args.train_biom)
    n_input = table.shape[0]

    sample_metadata = pd.read_table(args.sample_metadata, dtype=str)
    sample_metadata = sample_metadata.set_index(sample_metadata.columns[0])
    sample_metadata = sample_metadata.loc[table.ids()]

    model = TripletVAE(
        vae_model, batch_model, n_input, n_hidden=args.n_hidden,
        dropout=args.dropout, bias=args.bias, batch_norm=args.batch_norm,
        learning_rate=args.learning_rate,
        scheduler=args.scheduler)

    print(args)
    print(pytorch_lightning.__version__)
    print(model)
    # TODO: this checkpointing appears to be broken
    # it may have to do with the save_hyperparameters issue
    ckpt_path = os.path.join(
        args.output_directory,
        "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        period=1,
        monitor='val_loss',
        mode='min',
        verbose=True)

    if args.profile:
        profiler = AdvancedProfiler()
    else:
        profiler = None

    os.mkdir(args.output_directory)
    tb_logger = pl_loggers.TensorBoardLogger(f'{args.output_directory}/logs/')
    # save hyper-parameters to yaml file
    with open(f'{args.output_directory}/hparams.yaml', 'w') as outfile:
        yaml.dump(model._hparams, outfile, default_flow_style=False)

    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        check_val_every_n_epoch=1,
        gradient_clip_val=args.grad_clip,
        profiler=profiler,
        logger=tb_logger,
        callbacks=[checkpoint_callback])

    trainer.fit(model=model, datamodule=dm)
    trainer.save_checkpoint(
        args.output_directory + '/last_ckpt.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser = TripletVAE.add_model_specific_args(parser, add_help=False)
    parser = add_data_specific_args(parser, add_help=False)
    parser.add_argument(
        '--vae-model-path', help='Path of the VAE model.',
        required=False, type=str, default='')
    parser.add_argument(
        '--batch-model-path', help='Path of the batch classifier.',
        required=False, type=str, default='')
    args = parser.parse_args()
    main(args)
