import os
import argparse
import numpy as np
import torch
from catvae.trainer import MultVAE, add_data_specific_args
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler
from biom import load_table


def main(args):
    n_input = load_table(args.val_biom).shape[0]
    model = MultVAE(n_input,
                    n_latent=args.n_latent,
                    n_hidden=args.n_hidden,
                    basis=args.basis,
                    dropout=args.dropout,
                    bias=args.bias,
                    batch_norm=args.batch_norm,
                    encoder_depth=args.encoder_depth,
                    learning_rate=args.learning_rate,
                    scheduler=args.scheduler,
                    transform=args.transform)
    if args.load_from_checkpoint is not None:
        checkpoint = torch.load(
            args.load_from_checkpoint,
            map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
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
        args.train_biom, args.test_biom, args.valid_biom,
        metadata=args.sample_metadata, batch_category=args.batch_category,
        batch_size=args.batch_size, num_workers=args.num_workers)
    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        check_val_every_n_epoch=1,
        gradient_clip_val=args.grad_clip,
        profiler=profiler)
    ckpt_path = os.path.join(
        args.output_directory,
        trainer.logger.name,
        f"linear_vae_version_{trainer.logger.version}",
        "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        period=1,
        monitor='val_loss',
        mode='min',
        verbose=True)
    trainer.checkpoint_callback = checkpoint_callback
    trainer.fit(model, dm)
    torch.save(model.state_dict(),
               args.output_directory + '/last_ckpt.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser = MultVAE.add_model_specific_args(parser)
    parser = add_data_specific_args(parser)
    args = parser.parse_args()
    main(args)
