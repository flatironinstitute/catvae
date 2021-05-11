import os
import argparse
import numpy as np
import torch
from catvae.trainer import LightningLinearVAE
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler


def main(args):
    model = LightningLinearVAE(args)
    if args.load_from_checkpoint is not None:
        model = LightningLinearVAE(args)
        checkpoint = torch.load(
            args.load_from_checkpoint,
            map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = LightningLinearVAE(args)
    print(args)
    print(model)

    if (args.eigvectors is not None and
        args.eigvalues is not None):
        eigvectors = np.loadtxt(args.eigvectors)
        eigvalues = np.loadtxt(args.eigvalues)
        model.set_eigs(eigvectors, eigvalues)
    if args.profile:
        profiler = AdvancedProfiler()
    else:
        profiler = None
    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        check_val_every_n_epoch=1,
        gradient_clip_val=args.grad_clip,
        profiler=profiler
    )
    ckpt_path = os.path.join(
        args.output_directory,
        trainer.logger.name,
        f"linear_vae_version_{trainer.logger.version}",
        "checkpoints",
    )
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        period=1,
        monitor='val_loss',
        mode='min',
        verbose=True
    )
    trainer.checkpoint_callback = checkpoint_callback

    trainer.fit(model)
    torch.save(model.state_dict(),
               args.output_directory + '/last_ckpt.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser = LightningLinearVAE.add_model_specific_args(parser)
    parser.add_argument('--num-workers', type=int)
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
    args = parser.parse_args()
    main(args)
