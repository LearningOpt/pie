OUTPUT_DIR=${OUTPUT_DIR:-"saved_models/code_opt"}
BASE_MODEL=${BASE_MODEL:-"codellama/CodeLlama-7b-hf"}

torchrun --nproc_per_node=8 \
    --master_port=1234 full_ft.py \
    --base_model $BASE_MODEL \
    --data_path ./data/ \
    --output_dir $OUTPUT_DIR \
    --batch_size 32 \
    --micro_batch_size 2 \
    --num_epochs 1 \
    --learning_rate 1e-5 \
    --cutoff_len 2000 \
    --train_on_inputs False \
    --prompt_template_name "code_opt" \
    --use_flash_attention True \
    --train_name "train.jsonl" \
    --val_name "val.jsonl" \
    --test_name "test.jsonl" \
    --wandb_project "code_opt" \

# Copy tokenizer files to appropriate location, modify this if model is different
if [[ $BASE_MODEL == *"7b"* ]]; then
    cp -r ./tokenizer_files/7B/* $OUTPUT_DIR
elif [[ $BASE_MODEL == *"13b"* ]]; then
    cp -r ./tokenizer_files/13B/* $OUTPUT_DIR
else
    echo "Base model size not recognized. Tokenizer files not copied."
fi