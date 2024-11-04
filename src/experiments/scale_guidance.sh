#!/bin/bash

# Path to the prompts file
PROMPTS_FILE="src/experiments/scale_guidance_txt2image.txt"

# Read prompts into an array
PROMPTS=()
while IFS= read -r line || [[ -n "$line" ]]; do
    PROMPTS+=("$line")
done < "$PROMPTS_FILE"

# Outer loop for varying scale from 1 to 15 (steps of 3)
for SCALE in $(seq 1 3 15); do
    # Set the output directory based on the current scale
    SCALE_DIR="./output/experiment_scale/scale_${SCALE}"

    # Create the output directory for the current scale
    mkdir -p "${SCALE_DIR}"

    # Inner loop to iterate over each prompt with an increasing index
    INDEX=1
    for PROMPT in "${PROMPTS[@]}"; do
        # Create directory for the current index inside scale directory
        INDEX_DIR="${SCALE_DIR}/index_${INDEX}"
        mkdir -p "${INDEX_DIR}"

        # Run the script with the current prompt and scale
        python src/generate.py --steps 75 --outdir "${INDEX_DIR}" --prompt "${PROMPT}" --n-iter 8 --scale "${SCALE}"

        # Increment index for the next prompt
        ((INDEX++))
    done
done
