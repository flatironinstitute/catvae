import argparse
import numpy as np
from catvae.trainer import LightningCatVAE


def main(args):
    model = LightningCatVAE(args)
    if (args.eigvectors is not None and
        args.eigvalues is not None):
        eigvectors = np.load(args.eigvectors)
        eigvalues = np.load(args.eigvalues)
        model.set_eigs(eigvectors, eigvalues)
    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser = LightningCatVAE.add_model_specific_args(parser)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--eigvalues', type=str, default=None,
                        help='Ground truth eigenvalues (optional)', required=False)
    parser.add_argument('--eigvectors', type=str, default=None,
                        help='Ground truth eigenvectors (optional)', required=False)
    args = parser.parse_args()
    main(args)
