# Learning Performance-Improving Code Edits

Repository for *Learning Performance-Improving Code Edits* ([paper](https://openreview.net/forum?id=ix7rLVHXyY), [website](https://pie4perf.com/)).

🚨 Benchmarking programs is easy; but benchmarking programs in a reproducible and deterministic manner is very hard. 

🚨 LLMs are not great at program optimization out-of-the-box (at least for competitive programming problems)


We perform extensive experiments to evaluate and improve Large Language Models (LLMs) for program optimization. We built a custom evaluation framework that benchmarks program execution time in a highly-reliable manner and we provide a dataset annotated with execution time information from our environment. 

When measuring average program speedup, we obtained a fine-tuned version of CodeLlama13B that outperforms GPT4 and the best human programmer. Using self-play for program optimization, we also obtain a fine-tuned version of GPT3.5 that is even stronger. 


<!-- <img width="879" alt="image" src="https://raw.githubusercontent.com/madaan/pie-perf/main/docs/static/images/mainfig-v4.jpg"> 
-->

<img src="./docs/images/46_animation.gif" alt="Description of animation">


<!-- ## Updates 📢
[May 2023] A large number of problem statements in codenet were in Japanese. We have translated them to English using ChatGPT/GPT-4. The files are located [here](data/problem_statements_translated.zip) -->


## Dataset

- PIE is based on [IBM CodeNet](https://github.com/IBM/Project_CodeNet). Huge thanks to the authors of CodeNet for making their curated dataset available!

Our Train/Val/Test splits are located [here](https://drive.google.com/drive/folders/1E_yFqM8khN1HAH03OKhjheSlNI4rYTT7?usp=sharing). There is also a `train_with_synthetic.jsonl` file which contains and additional ~1.4K pairs generated via self-play. We also have subsets `train_hq_only.jsonl` and `train_hq_and_synthetic.jsonl` which contain only high-quality pairs and high-quality pairs + synthetic pairs respectively.

Testcases: 

- [Merged test cases](https://drive.google.com/file/d/1evBDJapwRvCQK6VUCTV8ZE9WG2k3QJQr/view?usp=sharing) containing both public and generated test cases: these test cases were the ones used for experiments in the paper.
- [Public test cases](https://drive.google.com/file/d/1RcUpZMOR8L2xYYWDZx7I0tHFzFgg7COO/view?usp=share_link). These test cases are sourced from IBM CodeNet.
- [Generated test cases](https://drive.google.com/file/d/1migwX4wpED0gDDxn7gS6q55vWeXIDgId/view?usp=drive_link). These test cases are sourced from [alphacode](https://github.com/google-deepmind/code_contests).

The column `tests` in the jsonl files will contain the indices which should be used for benchmarking models. 

##  Program Benchmarking with gem5

Benchmarking programs is easy; but benchmarking programs in a reproducible and deterministic manner is <i>very hard</i>.
It is important, because we want to compare the performance of different models on the same set of programs irrespective of 
a reserarcher's server configuration. Moreover, you can even wind up in scenarios where you can benchmark the same exact program and accidentally believe one is much faster than the other. 

<img src="./docs/images/arch.png" alt="gem5">

We built a custom evaluation framework that benchmarks program execution time in a highly-reliable manner.
We built an execution sandbox based on the <a href="https://www.gem5.org/">gem5</a> simulator. Given program termination/a program not timing out, benchmarking results are deterministic.
For our experiments, we use <a href="https://github.com/darchr/gem5-skylake-config">a simulated CPU of the Intel Skylake CPU</a>.
We provide an easy-to-use docker image and API that can be used to reproduce our results and for other researchers to continue to use for program optimization research.

Building the environment is similar to the [gym](https://github.com/Farama-Foundation/Gymnasium) API for reinforcement learning. After importing the module and running make, the docker image should automatically be pulled on the first iteration and a container created. The environment then provides a convenient abstraction for interacting with the environment. More information is located at [gem5](./gem5/README.md).

It is possible that on a separate architecture, the gem5 simulator runs slower or faster then when we ran it, so results could be influenced by more-frequent and less-frequent timeouts. Generally this should affect programs on the threshold of timing out, 
and it should affect more-aggressive optimizations (often "better" models) less than less-aggressive optimizations. 

```python
import simulator 

# pulls the image from docker hub if it doesn't exist and sets up a connection with a running container
env = simulator.make(arch='X86-skylake', optimization_flag='-O3')
# example sending a program to benchmark within the environment
gem5_benchmarking_results = env.submit_single_submission(...)
```
## Performance-Conditioning

Programs can typically be written in many ways with different performance profiles. When training a model to predict performance-improving edits with
a large dataset, it may be trained on a mix of large and small improvements, without any information on which improvements are more desirable than others. We introduce performance tags during training by associating each “fast” program with a tag indicating the optimal achievable performance across all solutions in the dataset. 

<img src="./docs/images/performance_conditioning.png" alt="perf-conditioning">

Specifically, the tag indicates how close that program is to peak performance on a binned-scale
{1, 2, . . . , 10}. Then at test time, we prompt the model with a test input and a maximal score tag “10/10”, directing it to generate a highly-optimized solution.

The performance tags are available for the [training dataset](#dataset) and can be used to train models with performance-conditioning. We also provide our fine-tuning code which adds the prompts during training and inference. 

## Self-Play

In an attempt to boost the performance of our models, we also investigate the use of self-play for program optimization as a data augmentation technique. Because there is a limited set of programs in our dataset, we use an off-the-shelf language model to generate new programs and a high-quality fine-tuned model to generate new optimizations. After taking some rigorous steps to ensure the generated programs are semantically novel and the optimizaitons are non-trivial, we use the generated programs and optimizations to create new program optimization pairs. 


The self-play notion comes from the fact that one model is used to generate the programs to solve and the other model is used to generate solve/propose the optimizations.

<!-- <img src="static/images_2/self_play.drawio.png" alt="Architecture" style="width: 100%; height: auto; object-fit: contain;"> -->

<img src="./docs/images/self_play.drawio.png" alt="self-play">

Our best model without self-play was with GPT3.5 turbo, our best fine-tuned model was trained with 4,085 high quality pairs. We were able to sample 3,314 novel programs and obtain 1,485 high-quality optimizations.

Using these additional 1,485 optimizations helped improve the performance of our fine-tuned model. We also performed an ablation by adding 1,485 next-best programs from the PIE dataset for fine-tuning GPT3.5 turbo, but these pairs led to performance degradation.

<!-- the script is at ./src/data_augmentation/data_augmentation_driver_final.sh -->
We provide our scripts for [sampling programs and detecting semantic duplicates](./data_augmentation/data_augmentation_driver_final.sh) and the [self-play data itself](#dataset).


# Running Experiments 

## Finetuning Open Source Models 

We provide a docker image at ```yimengzeng/pie:torch201``` which contains all of the dependencies for finetuning the model, you can also refer to ```docker/Dockerfile``` for the specific packages required to replicate the environment.

To finetune codellama with the entire PIE dataset and the non-performance-conditioned prompt, run
```bash
bash finetuning/train.sh
```
To finetune codellama with the performance-conditioned prompt, change the ```--prompt_template_name``` flag to ```"code_opt_w_speedup_pctile"``` More details are located in the ```finetuning``` directory.

## Finetuning OpenAI Models 

The script `openai_finetuning/finetune_openai.py` was used to finetune GPT3.5 Turbo. Its usage is as follows:

```bash
python finetune_openai.py PATH_TO_CONFIG.yaml
```

More details and an example config file are located in the `openai_finetuning` directory.

## Dynamic Retrieval

A notebook that can be used to prepare the retrieval dataset is `retrieval/retrieval.ipynb`. Given a training dataset and the test set examples to optimize, it will retrieve the K most similar training examples pairs for the given test set examples. The retrieved pairs are then used to prompt the model for optimized outputs.

## Sampling from Models

To generate prompts for the models, please follow details in the paper. Additional utilities for constructing prompts are located in `finetinung/templates` and the `funetuning/utils/prompter.py` module which constructs prompts. 

Samples from our fine-tuned models are located [here](https://drive.google.com/drive/folders/1criq4bpLlIaINzhjUAB18NZwDtEkk0Rj?usp=sharing). 

#### Sampling from Open Source Models
To sample optimized programs using the finetuned model with the ```text-generation-inference``` tool, first replace the ```PATH_TO_MODEL``` field to the acutal path of the finetuned model in ```server.sh```, and then to serve the model, run
```bash
bash finetuning/server.sh
```

To sample from the model just served with default parameters as in the paper, run
```bash
bash finetuning/sample.sh
```

More details are located in the ```finetuning``` directory.

#### Sampling from OpenAI 

We used [prompt-lib](https://github.com/reasoning-machines/prompt-lib/tree/main) to sample from OpenAI's endpoints. 

## Self-Play Experiments 

The directory `data_augmentation` contains the scripts used to sample and filter out novel competitive programming problems for PIE. 

Running ``data_augmentation/data_augmentation_driver_final.sh`` contains the final parameters we used to sample the problems. More details are located in the `data_augmentation` directory.

## Evaluation 

The evaluation driver is located in `gem5/gem5_eval.py`. This script requires a yaml configuration file to be passed in as an argument to `--config_path`. Example usage from the project directory would be: 

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python gem5/gem5_eval.py --config_path PATH_TO_EXPERIMENT_CONFIG.yaml
```

Results from our experiments can be located in [this google drive folder](https://drive.google.com/drive/folders/1criq4bpLlIaINzhjUAB18NZwDtEkk0Rj?usp=sharing).

More details are located in the `gem5` directory. 

----

## Citation

```
@inproceedings{pie_iclr_2024_spotlight,
    title={\href{https://openreview.net/pdf?id=ix7rLVHXyY}{Learning Performance-Improving Code Edits}},
    author={Shypula, Alexander and Madaan, Aman and Zeng, Yimeng and Alon, Uri and Gardner, Jacob and Hashemi, Milad and Neubig, Graham and Ranganathan, Parthasarathy and Bastani, Osbert and Yazdanbakhsh, Amir},
    booktitle={The Twelfth International Conference on Learning Representations (ICLR)},
    year={2024}
}
```


