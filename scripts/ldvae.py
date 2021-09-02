import argparse
import scvi
from catvae.trainer import MultVAE, add_data_specific_args
from biom import load_table
import yaml
import pandas as pd


# This is derived from the scvi notebook
# https://github.com/YosefLab/scvi-tutorials/blob/master/linear_decoder.ipynb
# Note that this is not going to be supported in the future
# so use at your own risk
def main(args):
    print(args)
    output_dir = args.output_directory
    epochs = args.epochs
    lr = args.learning_rate
    # Need to hack in sample metadata
    metadata = pd.read_table(args.sample_metadata, dtype=str)
    index_name = metadata.columns[0]
    metadata = metadata.set_index(index_name)

    train_biom = load_table(args.train_biom)
    # test_biom = load_table(args.test_biom)
    valid_biom = load_table(args.val_biom)
    # looks like we can't specify splits
    # so we'll just combined train/validate tables
    # this could give scvi a slight advantage, but whatev
    t = train_biom.merge(valid_biom)
    D, _ = t.shape

    obs_md = {i: {'taxonomy': 'None'} for i in t.ids(axis='observation')}
    if args.batch_category is not None:
        batch_cats = metadata.loc[t.ids(), args.batch_category].values
        sample_md = {i: {'batch': v} for i, v in zip(t.ids(), batch_cats)}
    else:
        sample_md = {i: {'batch': 'None'} for i in t.ids()}
    t.add_metadata(sample_md, axis='sample')
    t.add_metadata(obs_md, axis='observation')

    # careful here, this requires at least biom 2.1.10
    # https://github.com/biocore/biom-format/pull/845
    adata = t.to_anndata()
    adata.layers["counts"] = adata.X.copy()  # preserve counts
    if args.batch_category is not None:
        scvi.data.setup_anndata(adata, layer="counts", batch_key="batch")
    else:
        scvi.data.setup_anndata(adata, layer="counts")
    model = scvi.model.LinearSCVI(
        adata, dropout_rate=args.dropout,
        n_latent=args.n_latent, n_layers=args.encoder_depth,
        n_hidden=args.n_hidden)
    print(model)

    vargs = argparse.Namespace()
    hparams = vars(vargs)
    with open(f'{output_dir}/hparams.yaml', 'w') as outfile:
        yaml.dump(hparams, outfile, default_flow_style=False)

    model.train(max_epochs=epochs,
                plan_kwargs={'lr': lr},
                check_val_every_n_epoch=50)

    path = f'{output_dir}/last_ckpt.pt'
    model.save(path, save_anndata=True)
    # this model can be loaded via
    # model = scvi.model.LinearSCVI.load(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser = MultVAE.add_model_specific_args(parser, add_help=False)
    parser = add_data_specific_args(parser, add_help=False)
    args = parser.parse_args()
    main(args)
