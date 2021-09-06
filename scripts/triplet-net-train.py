import os
import argparse
import numpy as np
from catvae.trainer import MultVAE, TripletVAE
from catvae.trainer import BiomDataModule, add_data_specific_args
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning import loggers as pl_loggers
from biom import load_table
from skbio import TreeNode
import yaml


def main(args):
    if args.load_from_checkpoint is None:
        raise ValueError('`load-from-checkpoint` should be specified.')
    vae_model = MultVAE.load_from_checkpoint(args.load_from_checkpoint)
    model = TripletVAE(
        vae_model,
        n_input=vae_model.vae.decoder.in_features,
        n_hidden=args.n_hidden, n_layers=args.n_layers,
        learning_rate=args.learning_rate,
        vae_learning_rate=args.vae_lr,
        scheduler='cosine'
    )
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
    parser = TripletVAE.add_model_specific_args(parser, add_help=False)
    parser = add_data_specific_args(parser, add_help=False)
    args = parser.parse_args()
    main(args)
