# FROM ubuntu:22.04 as base
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

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

# Location of stoix folder
ARG folder=/home/app/stoix

# Set working directory
WORKDIR ${folder}

# Copy all code to the container
COPY . .

RUN echo "Installing requirements..."
RUN pip install --quiet --upgrade pip setuptools wheel &&  \
    pip install -e .

# Need to use specific cuda versions for jax
ARG USE_CUDA=true
RUN if [ "$USE_CUDA" = true ] ; \
    then pip install "jax[cuda12]>=0.4.10" -f "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html" ; \ 
    fi
EXPOSE 6006
