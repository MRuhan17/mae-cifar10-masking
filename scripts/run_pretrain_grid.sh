#!/bin/bash

# run_pretrain_grid.sh
# Shell-style pseudocode for sweeping mask types and ratios

for MASK_TYPE in random block
do
    for MASK_RATIO in 0.4 0.6 0.75 0.9
    do
        echo "Running MAE pretraining with mask_type=$MASK_TYPE  mask_ratio=$MASK_RATIO"

        python train_mae.py \
            --dataset cifar10 \
            --mask_type $MASK_TYPE \
            --mask_ratio $MASK_RATIO \
            --epochs 100 \
            --batch_size 128 \
            --save_dir outputs/${MASK_TYPE}_${MASK_RATIO}
    done
done
