model='PATH_TO_MODEL' # 'codellama/CodeLlama-7b-hf' for example
volume=$PWD/saved_models/ # share a volume with the Docker container to avoid downloading weights every run
max_best_of=20 # max number of samples to generate in parallel

docker run -e NVIDIA_VISIBLE_DEVICES="0,1,2,3" --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest \
--model-id $model --max-best-of $max_best_of
