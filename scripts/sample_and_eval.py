import subprocess
import time
import logging
import sys
import yaml
import shutil
import os

def start_generation_container(model, volume, max_best_of, port=4242, startup_timeout=600):
    # command = f"docker run --detach --gpus all --shm-size 1g -p {port}:80 -v {volume}:/data ghcr.io/huggingface/text-generation-inference:latest --model-id {model} --max-best-of {max_best_of}"
    # with 1,2,3,4,5,6,7 gpus 
    if not model.startswith("codellama"):
        model = f"data/{model}"
    # the first command may be 
    command = f"docker run --detach --gpus 1,2,3,4,5,6,7 --shm-size 1g -p {port}:80 -v {volume}:/data ghcr.io/huggingface/text-generation-inference:latest --model-id {model} --max-best-of {max_best_of}"
    # use the following line for podman or potentially for a different docker installation, the nvidia-docker command may vary 
    # command = f"docker run --detach -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 --shm-size 1g -p {port}:80 -v {volume}:/data ghcr.io/huggingface/text-generation-inference:latest --model-id {model} --max-best-of {max_best_of}"
    container_id = subprocess.check_output(command, shell=True).decode().strip()
    # wait until the logs say Connected
    while True:
        logging.info(f"Waiting for container to start with id {container_id} and timeout {startup_timeout} left")
        logs = subprocess.check_output(f"docker logs {container_id}", shell=True).decode()
        if "Connected" in logs:
            break
        time.sleep(5)
        startup_timeout -= 5
        if startup_timeout <= 0:
            raise TimeoutError("Timeout waiting for container to start")
    return container_id

def stop_generation_container(container_id):
    subprocess.run(f"docker stop {container_id}", shell=True)

def remove_generation_container(container_id):
    subprocess.run(f"docker rm {container_id}", shell=True)
    

def sample_from_container(test_file, output_file, do_sample, num_samples=8, max_new_tokens=1000, temperature=0.7, num_threads=20, prompt_name="code_opt"): 
    logging.info(f"Sampling from container with test_file {test_file} and output_file {output_file}")
    command = f"python finetuning/sample.py --test_file {test_file} --output_file {output_file} --do_sample {do_sample} --num_samples {num_samples} --max_new_tokens {max_new_tokens} --temperature {temperature} --num_threads {num_threads} --prompt_name {prompt_name}"
    logging.info(f"Running command {command}")
    p = subprocess.run(command, shell=True)
    logging.info(f"sample.py returned with code {p.returncode}")
    return p.returncode

def run_eval(eval_args):
    eval_args["model_generated_outputs_path"] = sampling_args["output_file"]
    eval_output_dir = eval_args["output_dir"]
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    else: 
        logging.warning(f"Output directory {eval_output_dir} already exists, overwriting")
    with open(os.path.join(eval_output_dir, "config.yaml"), "w") as f:
        yaml.dump(eval_args, f)
    logging.info(f"Running eval with args {eval_args}")
    cmd = f"python gem5/gem5_eval.py --config_path {os.path.join(eval_output_dir, 'config.yaml')}"
    logging.info(f"Running command {cmd}")
    p = subprocess.run(cmd, shell=True)
    logging.info(f"gem5_eval.py returned with code {p.returncode}")
    logging.info("Done")
    

def main(): 
    cfg_path = sys.argv[1]
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f)
    text_gen_args = cfg["text_gen_args"]
    sampling_args = cfg["sampling_args"]
    eval_args = cfg["eval_args"]

    # Check if the output directory for evaluation exists
    if os.path.exists(eval_args['output_dir']):
        logging.info(f"Output directory {eval_args['output_dir']} already exists. Skipping the entire script.")
        return

    # Check if the output file from sampling exists
    if os.path.exists(sampling_args['output_file']):
        logging.info(f"Output file {sampling_args['output_file']} from sampling already exists. Skipping container startup and sampling.")
    else:
        # Start the container and perform sampling
        logging.info(f"Starting generation container with args {text_gen_args}")
        container_id = start_generation_container(text_gen_args["generation_model_name"], text_gen_args["volume_mount"], text_gen_args["max_best_of"], port=text_gen_args["port"])
        logging.info(f"Sampling from container with args {sampling_args}")
        sample_from_container(**sampling_args)
        # Stop and remove the container
        logging.info(f"Stopping container with id {container_id}")
        stop_generation_container(container_id)
        logging.info(f"Removing container with id {container_id}")
        remove_generation_container(container_id)
        logging.info("Successfully removed container")

    # Run evaluation
    logging.info(f"Setting model_generated_outputs_path to {sampling_args['output_file']} and running eval with args {eval_args}")
    run_eval(eval_args)

    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    