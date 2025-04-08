#!/bin/bash
#SBATCH --job-name=8gpu_test
#SBATCH -o job.%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=8

# 加载环境
module load anaconda
module load cuda/12.1
#conda activate pytorch

# 运行测试程序
python test.py