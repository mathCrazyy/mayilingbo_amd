#!/usr/bin/env bash
# 用法: stapler_seedcheck.sh <PORT> <TAG>
PORT=$1; TAG=$2
cd /RoboTwin
export HIP_VISIBLE_DEVICES=0 ROBOTWIN_DISABLE_CUROBO=1 ROBOTWIN_EE_PLANNER=mplib PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
OUT=/workspace/runtime/outputs/stapler_seedcheck
for ST in 200000 300000 400000; do
  T0=$(date +%s)
  env -u ROBOTWIN_EXPERT_CACHE /opt/robotwin-env/bin/python scripts/eval_policy_xpolicylab.py \
    --task_name press_stapler --task_config demo_clean --policy_name LingBot-VLA-v2 \
    --protocol lingbot_vla_v2 --host 127.0.0.1 --port $PORT --device_id 0 --seed 0 \
    --seed_start $ST \
    --test_num 10 --expert_check true --accept_expert_info_on_failure true \
    --episode_info_output $OUT/info_${TAG}_s${ST}.jsonl \
    --eval_batch false --additional_info eval_video_log=false \
    > $OUT/${TAG}_s${ST}.log 2>&1
  W=$(( $(date +%s) - T0 ))
  rate=$(tr "\r" "\n" < $OUT/${TAG}_s${ST}.log | grep -a "Final success rate" | tail -1)
  echo "${TAG} seed=${ST} wall=${W}s ${rate}" >> $OUT/summary.txt
done
echo "${TAG} ALL DONE" >> $OUT/summary.txt
