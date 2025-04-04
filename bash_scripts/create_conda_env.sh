#!/bin/bash

# Check if the accelerator argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <accelerator>"
  echo "Available options for <accelerator>: tpu, gpu, cpu"
  exit 1
fi

ACCELERATOR=$1

# Activate Conda
source ~/miniconda3/etc/profile.d/conda.sh

# Create a new Conda environment
conda create -n stx python=3.10 -y

# Activate the environment
conda activate stx

# Install packages with pip
pip install -e .

# Install JAX with the appropriate accelerator support
if [ "$ACCELERATOR" == "tpu" ]; then
  pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
elif [ "$ACCELERATOR" == "gpu" ]; then
  pip install -U "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
elif [ "$ACCELERATOR" == "cpu" ]; then
  pip install -U jax
else
  echo "Invalid accelerator type: $ACCELERATOR"
  echo "Available options: tpu, gpu, cpu"
  exit 1
fi

echo "Installation complete with $ACCELERATOR support."
