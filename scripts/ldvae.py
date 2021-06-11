import os
import argparse
import numpy as np
import scvi
from catvae.trainer import MultVAE, BiomDataModule, add_data_specific_args
from biom import load_table
from skbio import TreeNode
import yaml


# This is derived from the scvi notebook
# https://github.com/YosefLab/scvi-tutorials/blob/master/linear_decoder.ipynb
# Note that this is not going to be supported in the future
# so use at your own risk
def main(args):
    print(args)
    train_biom = load_table(args.train_biom)
    test_biom = load_table(args.test_biom)
    valid_biom = load_table(args.val_biom)
    # looks like we can't specify splits
    # so we'll just combined train/validate tables
    # this could give scvi a slight advantage, but whatev
    t = train_biom.merge(valid_biom)
    # Need to hack in sample metadata
    D, _ = t.shape
    metadata = pd.read_table(metadata, dtype=str)
    index_name = metadata.columns[0]
    metadata = metadata.set_index(index_name)

    obs_md = [{'taxonomy': 'None'} for v in range(D)]
    sample_md = [{'batch': np.asscalar(v)}
                 for i, v in metadata.loc[t.ids(), args.batch_category]]

    # careful here, this requires at least biom 2.1.10
    # https://github.com/biocore/biom-format/pull/845
    adata = t.to_anndata()
    adata.layers["counts"] = adata.X.copy() # preserve counts
    scvi.data.setup_anndata(adata, layer="counts", batch_key="batch")

    model = scvi.model.LinearSCVI(
        adata, dropout_rate=args.dropout,
        n_latent=args.n_latent, n_layers=args.n_layers,
        n_hidden=args.n_hidden)
    print(model)
    model.train(max_epochs=args.epochs,
                plan_kwargs={'lr':args.learning_rate},
                check_val_every_n_epoch=50)

    args = argparse.Namespace()
    hparams = vars(args)
    os.mkdir(args.output_directory)
    with open(f'{args.output_directory}/hparams.yaml', 'w') as outfile:
        yaml.dump(hparams, outfile, default_flow_style=False)
    path = f'{args.output_directory}/{last_ckpt.pt}'
    model.save(path, save_anndata=True)
    # this model can be loaded via
    # model = scvi.model.LinearSCVI.load(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser = MultVAE.add_model_specific_args(parser, add_help=False)
    parser = add_data_specific_args(parser, add_help=False)
    args = parser.parse_args()
    main(args)
