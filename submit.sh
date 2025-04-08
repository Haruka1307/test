#!/bin/bash
#SBATCH --job-name=8gpu_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8          # 申请8卡
#SBATCH --cpus-per-task=64    # 8卡×8核=64核
#SBATCH --mem=2000G          # 8卡×250GB=2000GB
#SBATCH --time=00:10:00       # 运行10分钟

# 加载环境
module load anaconda
conda activate pytorch

# 运行测试程序
python 8gpu_test.py