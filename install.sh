#!/usr/bin/bash
set -e
CONDA_ENV_NAME=conda_env
echo "Starting installation in $(pwd)"
base_dir=$(pwd)
cd $base_dir
echo "Creating conda environment with prefix $(pwd)/$CONDA_ENV_NAME and installing packages"
conda env create -p $(pwd)/$CONDA_ENV_NAME
conda activate $(pwd)/$CONDA_ENV_NAME
conda install -c conda-forge matplotlib
conda install -c bioconda hhsuite
cd ${base_dir}/GUIFold3
pip install .
cd $base_dir
git clone https://github.com/google-deepmind/alphafold3.git
cd ${base_dir}/alphafold3
pip install .
cd $base_dir