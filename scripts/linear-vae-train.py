import os
import argparse
import numpy as np
import torch
from catvae.trainer import LightningLinearVAE
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def main(args):
    model = LightningLinearVAE(args)
    if (args.eigvectors is not None and
        args.eigvalues is not None):
        eigvectors = np.loadtxt(args.eigvectors)
        eigvalues = np.loadtxt(args.eigvalues)
        model.set_eigs(eigvectors, eigvalues)
    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        check_val_every_n_epoch=1,
        gradient_clip_val=args.grad_clip,
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
    parser.add_argument('--grad-clip', type=int, default=10)
    parser.add_argument('--eigvalues', type=str, default=None,
                        help='Ground truth eigenvalues (optional)', required=False)
    parser.add_argument('--eigvectors', type=str, default=None,
                        help='Ground truth eigenvectors (optional)', required=False)
    args = parser.parse_args()
    main(args)
