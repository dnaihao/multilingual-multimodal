#!/usr/bin/env bash

#SBATCH --job-name=visual_feature_extraction
#SBATCH --mail-user=dnaihao@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=mihalcea0
#SBATCH --partition=spgpu
#SBATCH --output=/home/%u/multilingual-multimodal/%x-%j.log

source /etc/profile.d/http_proxy.sh
bash emb_similarity.sh 
