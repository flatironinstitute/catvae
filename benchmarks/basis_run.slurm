#!/bin/bash
#
#SBATCH -N 1
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org

source ~/.bashrc
conda activate catvae

SIM=sparse_sim
# n_latent=10
# # Simulate the counts
# simulate-counts.py \
#        --latent-dim $n_latent \
#        --input-dim 200 \
#        --samples 1000 \
#        --depth 100 \
#        --output-dir $SIM


# SIM=dense_sim
# # Simulate the counts
# simulate-counts.py \
#        --latent-dim 10 \
#        --input-dim 100 \
#        --samples 1000 \
#        --depth 1000000 \
#        --output-dir $SIM
# SIM=dense_sim

# SIM=sparse_sim_1000
bases="$SIM/basis.nwk alr identity"
analytic='True False'
lr=1e-3
batch_size=100
epochs=1000

# Analytical Catvae
for basis in $bases
do
    OUT=catvae-analytic-$basis
    sbatch -p gpu --gres=gpu:v100-32gb:1 --wrap "catvae-train.py --num-workers 10 --gpus 1 --eigvalues $SIM/eigvals.txt --eigvectors $SIM/eigvecs.txt --basis $basis --learning-rate $lr --batch-size $batch_size --train-biom $SIM/train.biom --test-biom $SIM/test.biom --val-biom $SIM/valid.biom --bias False --steps-per-batch 100 --scheduler none --epochs $epochs --output-dir $OUT"

done

# Stochastic Catvae
for basis in $bases
do
    OUT=catvae-stochastic-$basis
    sbatch -p gpu --gres=gpu:v100-32gb:1 --wrap "linear-vae-train.py  --num-workers 10 --gpus 1 --eigvalues $SIM/eigvals.txt --eigvectors $SIM/eigvecs.txt --basis $basis --learning-rate $lr --batch-size $batch_size --use-analytic-elbo False --likelihood multinomial --train-biom $SIM/train.biom --test-biom $SIM/test.biom --val-biom $SIM/valid.biom --scheduler none --bias False --epochs $epochs --output-dir $OUT"
done
