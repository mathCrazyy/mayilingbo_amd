#!/usr/bin/env bash
cd /RoboTwin
export HIP_VISIBLE_DEVICES=0 ROBOTWIN_DISABLE_CUROBO=1 ROBOTWIN_EE_PLANNER=mplib PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export ROBOTWIN_EXPERT_CACHE=/workspace/runtime/outputs/exp_2x2/cache
OUT=/workspace/runtime/outputs/eps_scaling
mkdir -p $OUT
for N in 10 20 30 50 100; do
  T0=$(date +%s)
  /opt/robotwin-env/bin/python scripts/eval_policy_xpolicylab.py \
    --task_name adjust_bottle --task_config demo_clean --policy_name LingBot-VLA-v2 \
    --protocol lingbot_vla_v2 --host 127.0.0.1 --port 13400 --device_id 0 --seed 0 \
    --test_num $N --expert_check true --accept_expert_info_on_failure true \
    --episode_info_output $OUT/info_$N.jsonl \
    --eval_batch false --additional_info eval_video_log=false \
    > $OUT/run_$N.log 2>&1
  W=$(( $(date +%s) - T0 ))
  rate=$(tr "\r" "\n" < $OUT/run_$N.log | grep -a "Final success rate" | tail -1)
  ph=$(tr "\r" "\n" < $OUT/run_$N.log | grep -a "phase_timing" | tail -1 | grep -oE "expert_total=[0-9.]+s policy_total=[0-9.]+s")
  hits=$(grep -ac "expert_cache.*HIT" $OUT/run_$N.log)
  echo "N=$N wall=${W}s hits=$hits $rate $ph" >> $OUT/summary.txt
done
echo "SCALING DONE" >> $OUT/summary.txt
