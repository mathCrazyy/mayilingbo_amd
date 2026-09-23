#!/usr/bin/env bash
cd /RoboTwin
export HIP_VISIBLE_DEVICES=0 ROBOTWIN_DISABLE_CUROBO=1 ROBOTWIN_EE_PLANNER=mplib PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export ROBOTWIN_EXPERT_CACHE=/workspace/runtime/outputs/demo_clean_50x3_1gpu/episode_info
OUT=/workspace/runtime/outputs/determinism_test
for task in shake_bottle click_bell move_playingcard_away; do
  for rep in 1 2; do
    /opt/robotwin-env/bin/python scripts/eval_policy_xpolicylab.py \
      --task_name $task --task_config demo_clean --policy_name LingBot-VLA-v2 \
      --protocol lingbot_vla_v2 --host 127.0.0.1 --port 13400 --device_id 0 --seed 0 \
      --test_num 3 --expert_check true --accept_expert_info_on_failure true \
      --episode_info_output $OUT/${task}_rep${rep}_info.jsonl \
      --eval_batch false --additional_info eval_video_log=false \
      > $OUT/${task}_rep${rep}.log 2>&1
    r=$(tr "\r" "\n" < $OUT/${task}_rep${rep}.log | grep -a "Final success rate" | tail -1)
    echo "$task rep$rep: $r" >> $OUT/summary.txt
  done
done
echo "ALL DONE" >> $OUT/summary.txt
