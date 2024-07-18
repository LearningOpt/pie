# Gem5 Simulator for PIE

## Overview

This subdirectory contains the `gem5` module, which we use to interface with the `gem5` simulator. The `gem5` simulator is a full systema and CPU simulator that can be used to simulate the execution of a program on a computer system. We use `gem5` to simulate the execution of the programs in a determinstic and reproducible manner. 

For our experiments, we use <a href="https://github.com/darchr/gem5-skylake-config">a simulated CPU of the Intel Skylake CPU</a>.
We provide an easy-to-use docker image and API that can be used to reproduce our results and for other researchers to continue to use for program optimization research.

Building the environment is similar to the [gym](https://github.com/Farama-Foundation/Gymnasium) API for reinforcement learning. After importing the module and running make, the docker image should automatically be pulled on the first iteration and a container created. The environment then provides a convenient abstraction for interacting with the environment. 

Results from our experiments can be located in [this google drive folder](https://drive.google.com/drive/folders/1criq4bpLlIaINzhjUAB18NZwDtEkk0Rj?usp=sharing). 

<img src="../docs/images/arch.png" alt="gem5">

## Usage 
\***********************************************************************************************************************************

**Note that in order to use the module and its container for simulation, your architecture will need to be either x86-64 or Amd64** 

\***********************************************************************************************************************************

First you need to configure the pie project as part of your python path. You can do this by running the following command from the root of the pie project:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

On your system you will need to have docker installed. The module works using the Docker Python SDK and is designed to abstract away all the hassle of pulling the container and configuring the gem5 simulator. We have designed it to reflect the OpenAI Gym API, so it should be easy to use for anyone familiar with that.

```python

from gem5 import simulator 
env = simulator.make(...) 
results = env.submit_multiple_single_submissions(...)

```

In order to get started you will need the simulator.make() function to create an environment object which you can then use to submit to the simulator backend. 

#### Key Arguments for simulator.make()

- `arch`: The architecture to use. Currently only 'X86-skylake' is supported.
- `cpuset_cpus`: The cpus to use. If not specified, all cpus are used.
- `workers`: The number of workers to use. If not specified, all cpus are used.
- `gem5_acc_threshold`: If the functional accuracy is below this threshold, we skip any benchmarking and return the result early. 
- `port`: The port to use for communication.
- `optimization_flag`: The GCC optimization flag to use for compilation, for our work we used '-O3'.
- `cpu_type`: The type of CPU configuration to use. For our work we used 'Verbatim' from the skylake configuration used. 
- `timeout_seconds_gem5`: The timeout in seconds for the gem5 simulator, for our work we used 120 seconds for evaluation. 
- `verbose`: We highly recommend setting this to True to monitor the progress of the gem5 simulator.
- `exit_early_on_fail`: If True, we exit early if any individual test case times out or encounters a runtime error, we highly recommend this to be set to True for speeding things up if you're only evaluating, as we that would not contribute to any speedups. 

#### Key Arguments for env.submit_multiple_single_submissions()

- `code_list`: A list of strings, each string is the code of a single submission.
- `testcases_list`: Each sublist consists of the test cases used for benchmarking the corresponding code: these are the integer indices of the test cases in the test case pool.
- `problem_id_list`: A list of strings, each string is the problem id for the corresponding code.
- `timing_env`: The timing environment to use: currently only 'gem5' is supported, we have prototype support for hardware based benchmarking on your machine using 'hyperfine' or 'both' but the 'hyperfine' support is not fully implemented yet. 

## Evaluation Script

The evaluation driver is located in `gem5/gem5_eval.py`. This script requires a yaml configuration file to be passed in as an argument to `--config_path`. Example usage from the project directory would be: 

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python gem5/gem5_eval.py --config_path PATH_TO_EXPERIMENT_CONFIG.yaml
```

The yaml configuration file should contain at least the following fields:

- `model_generated_outputs_path`: The path to the model generated outputs. This should be a `.jsonl` file containing the model generated outputs in addition to all other metadata in the test set file. 
- `output_dir`: The directory to output the results to.
- `reference_file_path`: The path to the reference file. This should be the reference `.jsonl` file containing the reference outputs in addition to all other metadata in the test set file.
- `model_generated_potentially_faster_code_col`: The column in the model generated outputs that contains the model's generations of potentially faster code. We've used "generated_answers" as a default.

An example is provided in [gem5/template_config.yaml](template_config.yaml).
