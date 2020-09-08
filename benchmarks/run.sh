SIM=test

# Simulate the counts
simulate-counts.py \
       --latent-dim 10 \
       --input-dim 20 \
       --samples 1000 \
       --depth 100000 \
       --output-dir $SIM

bases="$SIM/basis.nwk alr identity"
analytic='True False'
lr=1e-3
batch_size=100
epochs=2
# Analytical Catvae
for basis in $bases
do
    OUT=catvae-analytic-$basis
    catvae-train.py \
        --num-workers 30 \
        --gpus 1 \
        --eigvalues $SIM/eigvals.txt \
        --eigvectors $SIM/eigvecs.txt \
        --basis $basis \
        --learning-rate $lr \
        --batch-size $batch_size \
        --train-biom $SIM/train.biom \
        --test-biom $SIM/test.biom \
        --val-biom $SIM/valid.biom \
        --steps-per-batch 10 \
        --epochs $epochs \
        --output-dir $OUT

done

# Stochastic Catvae
for basis in $bases
do
    OUT=catvae-stochastic-$basis
    linear-vae-train.py \
        --num-workers 30 \
        --gpus 1 \
        --eigvalues $SIM/eigvals.txt \
        --eigvectors $SIM/eigvecs.txt \
        --basis $basis \
        --learning-rate $lr \
        --batch-size $batch_size \
        --use-analytic-elbo False \
        --likelihood multinomial \
        --train-biom $SIM/train.biom \
        --test-biom $SIM/test.biom \
        --val-biom $SIM/valid.biom \
        --epochs $epochs \
        --output-dir $OUT
done

# Gaussian linear VAE
for basis in $bases
do
    OUT=linear-vae-analytic-$basis
    linear-vae-train.py \
        --num-workers 30 \
        --gpus 1 \
        --eigvalues $SIM/eigvals.txt \
        --eigvectors $SIM/eigvecs.txt \
        --basis $basis \
        --learning-rate $lr \
        --batch-size $batch_size \
        --use-analytic-elbo True \
        --likelihood gaussian \
        --train-biom $SIM/train.biom \
        --test-biom $SIM/test.biom \
        --val-biom $SIM/valid.biom \
        --epochs $epochs \
        --output-dir $OUT
done

for basis in $bases
do
    OUT=linear-vae-stochastic-$basis
    linear-vae-train.py \
        --num-workers 30 \
        --gpus 1 \
        --eigvalues $SIM/eigvals.txt \
        --eigvectors $SIM/eigvecs.txt \
        --basis $basis \
        --learning-rate $lr \
        --batch-size $batch_size \
        --use-analytic-elbo False \
        --likelihood gaussian \
        --train-biom $SIM/train.biom \
        --test-biom $SIM/test.biom \
        --val-biom $SIM/valid.biom \
        --epochs $epochs \
        --output-dir $OUT
done
