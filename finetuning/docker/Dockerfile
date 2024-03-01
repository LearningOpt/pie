FROM docker.io/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-get update && apt-get install -y wget git
RUN pip install tqdm wandb lightning fire rotary_embedding_torch pandas numpy matplotlib transformers datasets peft text_generation ninja packaging
RUN MAX_JOBS=32 pip install flash-attn --no-build-isolation
RUN pip install accelerate bitsandbytes
RUN pip install scipy
RUN pip install sentencepiece

# ENV WANDB_API_KEY=yourkeyhere
WORKDIR /workspace