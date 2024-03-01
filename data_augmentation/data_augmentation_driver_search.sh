#!/bin/bash

PATH_TO_PIE=""
WORKING_DIR=""
RESULTS_DIR=""
TIMEOUT=5
MAX_TOKENS=16000
MAX_PROMPT_LENGTH=10000
MODEL="gpt-3.5-turbo-16k-0613"
GENERATION_STRATEGY="code_only"

PARAM_SETS=(
  "1.2 0.8 5 20"
  "1 0.8 5 20"
  "1 0.9 5 20"
  "0.8 1 5 20"
  "0.5 1 5 20"
  "1 0.8 1 100"
  "1 0.9 1 100"
  "1 1 1 100"
  "0.7 1 1 100"
  "0.5 1 1 100"
)


# Loop through each set of parameters and run the Python script
for PARAM_SET in "${PARAM_SETS[@]}"; do
  # Split the parameter set into individual variables
  read -r TEMPERATURE TOP_P NUM_SAMPLES TOTAL_ITERATIONS <<< "$PARAM_SET"
  
  echo "Running with Temperature=$TEMPERATURE, Top-p=$TOP_P, Num_Samples=$NUM_SAMPLES, Total_Iterations=$TOTAL_ITERATIONS"

  python3 $PATH_TO_PIE/src/data_augmentation/data_augmentation.py \
      --working_dir $WORKING_DIR \
      --results_dir_root $RESULTS_DIR \
      --timeout $TIMEOUT \
      --temperature $TEMPERATURE \
      --top_p $TOP_P \
      --max_tokens $MAX_TOKENS \
      --max_prompt_length $MAX_PROMPT_LENGTH \
      --model $MODEL \
      --generation_strategy $GENERATION_STRATEGY \
      --num_samples $NUM_SAMPLES \
      --total_iterations $TOTAL_ITERATIONS 2>&1 | tee  $RESULTS_DIR/temperature_${TEMPERATURE}_top_p_${TOP_P}_num_samples_${NUM_SAMPLES}_total_iterations_${TOTAL_ITERATIONS}.log
done:q
