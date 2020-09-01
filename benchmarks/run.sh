
# Simulate the counts
simulate-counts.py \
       --latent-dim 50 \
       --input-dim 1000 \
       --samples 10000 \
       --depth 100000 \
       --output-dir simulation

SIM=simulation
OUT=catvae-fit-analytic
catvae-train.py \
    --num-workers 30 \
    --gpus 1\
    --eigvalues $SIM/eigvals.txt \
    --eigvectors $SIM/eigvecs.txt \
    --basis $SIM/basis.nwk \
    --learning-rate 1e-3 \
    --batch-size 1000 \
    --train-biom $SIM/train.biom \
    --test-biom $SIM/test.biom \
    --val-biom $SIM/valid.biom \
    --epochs 100 \
    --output-dir $OUT

