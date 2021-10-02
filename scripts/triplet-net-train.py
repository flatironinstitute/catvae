import os
import argparse
import numpy as np
from catvae.trainer import MultBatchVAE, TripletVAE
from catvae.trainer import BiomDataModule, TripletDataModule
from catvae.trainer import add_data_specific_args
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning import loggers as pl_loggers
import yaml


def main(args):
    print(args)
    if args.load_from_checkpoint is None:
        raise ValueError('`load-from-checkpoint` should be specified.')

    model = TripletVAE(
        args.load_from_checkpoint,
        n_hidden=args.n_hidden, n_layers=args.n_layers,
        learning_rate=args.learning_rate,
        vae_learning_rate=args.vae_lr,
        scheduler=args.scheduler)

    print(model)
    if args.profile:
        profiler = AdvancedProfiler()
    else:
        profiler = None

    dm = TripletDataModule(
        args.train_biom, args.test_biom, args.val_biom,
        metadata=args.sample_metadata,
        batch_category=args.batch_category,
        class_category=args.class_category,
        segment_triples=args.segment_triples,
        batch_size=args.batch_size, num_workers=args.num_workers)
    ckpt_path = os.path.join(args.output_directory, "checkpoints")
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_path,
                                          period=1,
                                          monitor='val/triplet_loss',
                                          mode='min',
                                          verbose=True)

    os.mkdir(args.output_directory)
    tb_logger = pl_loggers.TensorBoardLogger(f'{args.output_directory}/logs/')
    # save hyper-parameters to yaml file
    with open(f'{args.output_directory}/hparams.yaml', 'w') as outfile:
        yaml.dump(model._hparams, outfile, default_flow_style=False)

    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        check_val_every_n_epoch=10,
        gradient_clip_val=args.grad_clip,
        profiler=profiler,
        logger=tb_logger,
        callbacks=[checkpoint_callback])

    trainer.fit(model, dm)
    ckpt_path = args.output_directory + '/last_ckpt.pt'
    trainer.save_checkpoint(ckpt_path)

    # Perform KNN classification
    batch = next(iter(dm.test_dataloader()))
    res = model.test_step(batch, 0)['test/knn_results']
    open(f'{args.output_directory}/cross_validation.csv', 'w').write(res)
    # unfortunately, this appears to be broken
    # dl = dm.predict_dataloader()
    # batch = next(iter(dl))
    # model.predict_step(batch, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser = TripletVAE.add_model_specific_args(parser, add_help=False)
    parser = add_data_specific_args(parser, add_help=False)
    args = parser.parse_args()
    main(args)
