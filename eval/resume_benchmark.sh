#!/usr/bin/env bash
# 断点续跑闭环评测（实例重建后执行一条命令即可）
cd /RoboTwin
exec /opt/robotwin-env/bin/python experiments/lingbot_vla_v2_6b_robotwin/scripts/run_clean_benchmark.py \
  --gpu-count 1 --task-config demo_clean --episodes 3 \
  --run-name demo_clean_50x3_1gpu --resume
