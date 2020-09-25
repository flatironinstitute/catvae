#!/bin/bash
#
#SBATCH -N 1
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org

source ~/.bashrc
conda activate catvae
SIM=sparse_sim

n_latent=10
# Simulate the counts
simulate-counts.py \
       --latent-dim $n_latent \
       --input-dim 200 \
       --samples 1000 \
       --depth 100 \
       --output-dir $SIM


#bases="$SIM/basis.nwk alr identity"
basis=$SIM/basis.nwk
analytic='True False'
layers="1 3 5"
lr=1e-3
batch_size=100
epochs=100
# Analytical Catvae
# for depth in $layers
# do
#     OUT=catvae-analytic-ilr-$depth-layers
#     sbatch -p gpu --gres=gpu:v100-32gb:1 --wrap "catvae-train.py --num-workers 10 --gpus 1 --eigvalues $SIM/eigvals.txt --eigvectors $SIM/eigvecs.txt --basis $basis --learning-rate $lr --batch-size $batch_size --train-biom $SIM/train.biom --test-biom $SIM/test.biom --val-biom $SIM/valid.biom --steps-per-batch 30 --encoder-depth $depth --n-latent $n_latent --epochs $epochs --output-dir $OUT"
# done

# Stochastic Catvae
for depth in $layers
do
    OUT=catvae-stochastic-ilr-$depth-layers
    sbatch -p gpu --gres=gpu:v100-32gb:1 --wrap "linear-vae-train.py  --num-workers 10 --gpus 1 --eigvalues $SIM/eigvals.txt --eigvectors $SIM/eigvecs.txt --basis $basis --learning-rate $lr --batch-size $batch_size --use-analytic-elbo False --likelihood multinomial --train-biom $SIM/train.biom --test-biom $SIM/test.biom --val-biom $SIM/valid.biom --encoder-depth $depth --n-latent $n_latent  --epochs $epochs --output-dir $OUT"
done
