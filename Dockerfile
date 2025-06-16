# FROM ubuntu:22.04 as base
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Ensure no installs try to launch interactive screen
ARG DEBIAN_FRONTEND=noninteractive

COPY --from=ghcr.io/astral-sh/uv:0.4.28 /uv /uvx /bin/

# Update packages and install python3.10 and other dependencies
RUN apt-get update -y && \
    apt-get install -y software-properties-common git && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.10 python3.10-dev python3-pip python3.10-venv && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 10 && \
    python -m venv stoix && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Location of stoix folder
ARG folder=/home/app/stoix

# Set working directory
WORKDIR ${folder}

# Copy all code to the container
COPY . .

RUN echo "Installing requirements..."
RUN uv sync

# Need to use specific cuda versions for jax
ARG USE_CUDA=true
RUN if [ "$USE_CUDA" = true ] ; \
    then pip install "jax[cuda12]>=0.4.25" -f "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html" ; \
    fi
EXPOSE 6006
