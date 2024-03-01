The script `finetune_openai.py` was used to finetune GPT3.5 Turbo. Its usage is as follows:

```bash
python finetune_openai.py PATH_TO_CONFIG.yaml
```

We've included a sample config file `config.yaml` in this directory. The config file should contain the following fields:

```yaml
api_key: "YOUR_OPENAI_API_KEY"
organization: "YOUR_OPENAI_ORGANIZATION (optional)"
input_train_path: "PATH_TO_TRAINING_DATA"
input_test_path: "PATH_TO_VALIDATION_DATA"
max_train: -1
max_val: -1
max_len: -1
epochs: NUMBER_OF_EPOCHS (we used 1)
output_dir: "PATH_TO_OUTPUT_DIR"
model_suffix: "SUFFIX_FOR_MODEL_NAME"
```
