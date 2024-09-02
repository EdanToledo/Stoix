#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Download Miniconda
CONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
CONDA_URL="https://repo.anaconda.com/miniconda/$CONDA_INSTALLER"

echo "Downloading Miniconda installer..."
if ! wget -q --show-progress "$CONDA_URL"; then
  echo "Error: Failed to download Miniconda installer."
  exit 1
fi

# Step 2: Install Miniconda (silently with default settings)
echo "Installing Miniconda..."
bash "$CONDA_INSTALLER" -b

# Clean up the installer after installation
rm -f "$CONDA_INSTALLER"

# Step 3: Initialize Conda (Assuming bash shell)
CONDA_PATH="$HOME/miniconda3"

# Ensure Conda is initialized on every new shell
if ! grep -q "source $CONDA_PATH/etc/profile.d/conda.sh" ~/.bashrc; then
  echo "source $CONDA_PATH/etc/profile.d/conda.sh" >> ~/.bashrc
  echo "Conda initialization added to .bashrc"
fi

# Apply the .bashrc changes to the current shell session
echo "Initializing Conda for the current shell..."
source "$CONDA_PATH/etc/profile.d/conda.sh"

echo "Miniconda installation and initialization complete."

# Optional: Update conda to the latest version
echo "Updating Conda to the latest version..."
conda update -n base -c defaults conda -y
