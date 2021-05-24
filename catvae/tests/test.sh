simulate-counts.py \
    --latent-dim 10 \
    --input-dim 50 \
    --samples 500 \
    --depth 10000 \
    --no-batch-effect \
    --output-dir simulation_directory

linear-vae-train.py \
    --basis simulation_directory/tree.nwk \
    --n-latent 10 \
    --n-hidden 10 \
    --dropout 0.5 \
    --no-bias \
    --no-batch-norm \
    --encoder-depth 1 \
    --learning-rate 1e-1 \
    --transform pseudocount \
    --scheduler cosine \
    --train-biom simulation_directory/train.biom \
    --test-biom simulation_directory/test.biom \
    --val-biom simulation_directory/valid.biom \
    --batch-size 50 \
    --epochs 2 \
    --num-workers 1 \
    --gpus 0 \
    --output-directory vae_results

# test to make sure that it can load from checkpoint
linear-vae-train.py \
    --load-from-checkpoint vae_results/last_ckpt.pt \
    --basis simulation_directory/tree.nwk \
    --n-latent 10 \
    --n-hidden 10 \
    --dropout 0.5 \
    --no-bias \
    --no-batch-norm \
    --encoder-depth 1 \
    --learning-rate 1e-1 \
    --transform pseudocount \
    --scheduler cosine \
    --train-biom simulation_directory/train.biom \
    --test-biom simulation_directory/test.biom \
    --val-biom simulation_directory/valid.biom \
    --batch-size 50 \
    --epochs 2 \
    --num-workers 1 \
    --gpus 0 \
    --output-directory vae_results_2

# clean up
rm -r vae_results
rm -r vae_results_2
rm -r simulation_directory

simulate-counts.py \
    --latent-dim 10 \
    --input-dim 50 \
    --samples 500 \
    --depth 10000 \
    --batch-effect \
    --batches 3 \
    --output-dir simulation_directory

linear-vae-batch-train.py \
    --basis tree.nwk \
    --n-latent 10 \
    --n-hidden 10 \
    --dropout 0.5 \
    --no-bias \
    --no-batch-norm \
    --batch-size 50 \
    --encoder-depth 1 \
    --learning-rate 1e-1 \
    --transform pseudocount \
    --scheduler cosine \
    --batch-category batch_category \
    --sample-metadata simulation_directory/metadata.txt \
    --batch-prior simulation_directory/batch_priors.txt \
    --train-biom simulation_directory/train.biom \
    --test-biom simulation_directory/test.biom \
    --val-biom simulation_directory/valid.biom \
    --epochs 3 \
    --num-workers 1 \
    --gpus 0 \
    --output-directory vae_results
# test checkpointing
linear-vae-batch-train.py \
    --load-from-checkpoint vae_results/last_ckpt.pt \
    --basis tree.nwk \
    --n-latent 10 \
    --n-hidden 10 \
    --dropout 0.5 \
    --no-bias \
    --no-batch-norm \
    --batch-size 50 \
    --encoder-depth 1 \
    --learning-rate 1e-1 \
    --transform pseudocount \
    --scheduler cosine \
    --batch-category batch_category \
    --sample-metadata simulation_directory/metadata.txt \
    --batch-prior simulation_directory/batch_priors.txt \
    --train-biom simulation_directory/train.biom \
    --test-biom simulation_directory/test.biom \
    --val-biom simulation_directory/valid.biom \
    --epochs 3 \
    --num-workers 1 \
    --gpus 0 \
    --output-directory vae_results_2

# clean up
rm -r vae_results
rm -r vae_results_2
rm -r simulation_directory

