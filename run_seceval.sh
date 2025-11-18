#!/bin/bash

# 激活 conda 环境
source /home/muzammal/miniconda3/etc/profile.d/conda.sh
conda activate vllm2

# 多进程相关设置
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LOGLEVEL=INFO

# 输出目录
mkdir -p /home/muzammal/Projects/lighteval/results

echo "Running LightEval..."

# 运行 LightEval，注意位置参数的顺序和格式
lighteval vllm \
  "model_name=Qwen/Qwen3-4B,dtype=bfloat16" \
  /home/muzammal/Projects/lighteval/examples/seceval.txt \  # community|seceval:mcqa|0|0
  --custom-tasks /home/muzammal/Projects/lighteval/community_tasks/sec2evalmcqa.py \
  --output-dir /home/muzammal/Projects/lighteval/results \
  --save-details
