#!/bin/bash

# run_probe_grid.sh
# Shell-style pseudocode for running linear probes on all pretrained MAE models

MASK_TYPES=("random" "block")
MASK_RATIOS=("0.4" "0.6" "0.75" "0.9")

for MASK_TYPE in "${MASK_TYPES[@]}"
do
    for MASK_RATIO in "${MASK_RATIOS[@]}"
    do
        echo "Running linear probe for $MASK_TYPE mask with ratio $MASK_RATIO"

        # Location of pretrained encoder
        ENCODER_PATH="results/pretrain/${MASK_TYPE}_${MASK_RATIO}/encoder.pth"

        # Where to store probe results
        OUTPUT_DIR="results/linear_probe/${MASK_TYPE}_${MASK_RATIO}"
        mkdir -p "$OUTPUT_DIR"

        # Train linear probe
        python src/train_linear_probe.py \
            --encoder_path "$ENCODER_PATH" \
            --batch_size 128 \
            --epochs 20 \
            --lr 1e-3 \
            --device cuda \
            > "$OUTPUT_DIR/log.txt"

        # Optional: save accuracy summary
        echo "Saved probe results to $OUTPUT_DIR"
    done
done
