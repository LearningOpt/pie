# Finetuning Scripts for codellama

## Overview

This subdirectory contains the scripts used to finetune codellama models for PIE. ``train.sh`` contains an example bash script for finetuning codellama-7b with the default prompt. ``sample.py`` and ``server.sh`` are used for sampling with the prompt templates and the [text-generation-inference](https://github.com/huggingface/text-generation-inference) API.

## Docker Setup for Finetuning

To use the provided Docker image for finetuning, you need to install Docker and mount the directory properly. Follow these steps:

1. Install Docker: Follow the instructions on the [official Docker website](https://docs.docker.com/get-docker/) to install Docker on your system.

2. Mount the directory for the data: When running the Docker container, use the `-v` option to mount the directory containing your data. For example:
   ```bash
   docker run -v /path/to/your/data:/workspace/data yimengzeng/pie:torch201
    ```

## Finetuning 

We provide a docker image at yimengzeng/pie:torch201 which contains all of the dependencies for finetuning the model, you can also refer to docker/Dockerfile for the specific packages required to replicate the environment.

To finetune codellama with the entire PIE dataset and the non-performance-conditioned prompt, run

```bash
bash train.sh
```

inside the Docker container.

To finetune codellama with the performance-conditioned prompt, change the --prompt_template_name flag to "code_opt_w_speedup_pctile".

To use different training files, modify the --train_name, --val_name, and --test_name flags in train.sh with the paths to your training, validation, and test files, respectively and mount the directory containing the files when running the Docker container.


## Sampling

To generate prompts for the models, please follow details in the paper. Additional utilities for constructing prompts are located in `templates` and the `utils/prompter.py` module which constructs prompts. 

To sample optimized programs using the finetuned model with the text-generation-inference tool, first replace the PATH_TO_MODEL field to the actual path of the finetuned model in server.sh, and then to serve the model, run

```bash
bash server.sh
```
To sample from the model just served with default parameters as in the paper, run

```bash
bash sample.sh
```
Note that sampling does not require you to spin up the container on your own. You can modify the following parameters in `server.sh` and `sample.sh`:

For `server.sh`:
- `model`: Set this to the path of your finetuned model, e.g., `'codellama/CodeLlama-7b-hf'`.
- `volume`: Set this to the path where your model is stored, e.g., `$PWD/saved_models/`.
- `max_best_of`: Set this to the maximum number of samples to generate in parallel, e.g., `20`.

For `sample.sh`:
- `--test_file`: Set this to the path of your test file.
- `--output_file`: Set this to the path where you want to save the results.
- `--num_samples`: Set this to the number of samples you want to generate.
- `--num_threads`: Set this to the number of threads you want to use for sampling.
- `--prompt_name`: Set this to the name of the prompt template you want to use.
- `--temperature`: Set this to the temperature parameter for sampling.


## Models

Here are links to the finetuned models used in the paper and the corresponding pre-trained models used for finetuning:

| Experiment | Model | Type | Pretrained Link | Finetuned Link |
|------------|-------|------|-----------------|----------------|
| All | CodeLlama 7B | FT | [codellama/CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf) | [LearningOpt/pie-all-uncon-7b](https://huggingface.co/LearningOpt/pie-all-uncon-7b) |
| All | CodeLlama 13B | FT | [codellama/CodeLlama-13b-hf](https://huggingface.co/codellama/CodeLlama-13b-hf) | [LearningOpt/pie-all-uncon-13b](https://huggingface.co/LearningOpt/pie-all-uncon-13b) |
| HQ | CodeLlama 7B | FT | [codellama/CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf) | [LearningOpt/pie-hq-selfplay-7b](https://huggingface.co/LearningOpt/pie-hq-selfplay-7b) |
| HQ | CodeLlama 13B | FT | [codellama/CodeLlama-13b-hf](https://huggingface.co/codellama/CodeLlama-13b-hf) | [LearningOpt/pie-hq-selfplay-13b](https://huggingface.co/LearningOpt/pie-hq-selfplay-13b) |
| All w/Perf-Cond | CodeLlama 7B | FT | [codellama/CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf) | [LearningOpt/pie-conditioned-7b](https://huggingface.co/LearningOpt/pie-conditioned-7b) |
| All w/Perf-Cond | CodeLlama 13B | FT | [codellama/CodeLlama-13b-hf](https://huggingface.co/codellama/CodeLlama-13b-hf) | [LearningOpt/pie-conditioned-13b](https://huggingface.co/LearningOpt/pie-conditioned-13b) |
| HQ + Self-Play | CodeLlama 7B | FT | [codellama/CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf) | [LearningOpt/pie-hq-selfplay-7b](https://huggingface.co/LearningOpt/pie-hq-selfplay-7b) |
| HQ + Self-Play | CodeLlama 13B | FT | [codellama/CodeLlama-13b-hf](https://huggingface.co/codellama/CodeLlama-13b-hf) | [LearningOpt/pie-hq-selfplay-13b](https://huggingface.co/LearningOpt/pie-hq-selfplay-13b) |
