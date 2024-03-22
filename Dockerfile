# FROM ubuntu:22.04 as base
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Ensure no installs try to launch interactive screen
ARG DEBIAN_FRONTEND=noninteractive

# Update packages and install python3.10 and other dependencies
RUN apt-get update -y && \
    apt-get install -y software-properties-common git && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.10 python3.10-dev python3-pip python3.10-venv && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 10 && \
    python -m venv stoix && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setup virtual env and path
ENV VIRTUAL_ENV /stoix
ENV PATH /stoix/bin:$PATH

# Set working directory
WORKDIR /build

# Copy all code needed to install dependencies
COPY ./requirements ./requirements
COPY pyproject.toml ./pyproject.toml
COPY ./stoix ./stoix

# Need to use specific cuda versions for jax
ARG USE_CUDA=true

RUN pip install --quiet --upgrade pip setuptools wheel && \
    if [ "$USE_CUDA" = true ]; then \
        pip install "jax[cuda11_pip]<=0.4.13" -f "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"; \
    fi

# Install from pyproject.toml with verbose logging and ignore those packages already installed
RUN pip install --verbose --no-cache-dir --ignore-installed -e .

ENTRYPOINT [ "/stoix/bin/python" ]