#!/usr/bin/env bash
# 用法: exp_lane.sh <PORT> <MODEL_TAG>
PORT=$1; TAG=$2
TASKS="adjust_bottle blocks_ranking_rgb shake_bottle click_alarmclock press_stapler"
cd /RoboTwin
export HIP_VISIBLE_DEVICES=0 ROBOTWIN_DISABLE_CUROBO=1 ROBOTWIN_EE_PLANNER=mplib PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
EXP=/workspace/runtime/outputs/exp_2x2
R=$EXP/results
run_one() { # task cond(nocache|cache|cache2)
  t=$1; c=$2
  T0=$(date +%s)
  if [ "$c" = "nocache" ]; then
    env -u ROBOTWIN_EXPERT_CACHE /opt/robotwin-env/bin/python scripts/eval_policy_xpolicylab.py \
      --task_name $t --task_config demo_clean --policy_name LingBot-VLA-v2 \
      --protocol lingbot_vla_v2 --host 127.0.0.1 --port $PORT --device_id 0 --seed 0 \
      --test_num 10 --expert_check true --accept_expert_info_on_failure true \
      --episode_info_output $R/info_${TAG}_${t}_nc.jsonl \
      --eval_batch false --additional_info eval_video_log=false > $R/log_${TAG}_${t}_nc.log 2>&1
  else
    suffix=$([ "$c" = "cache2" ] && echo c2 || echo c)
    ROBOTWIN_EXPERT_CACHE=$EXP/cache /opt/robotwin-env/bin/python scripts/eval_policy_xpolicylab.py \
      --task_name $t --task_config demo_clean --policy_name LingBot-VLA-v2 \
      --protocol lingbot_vla_v2 --host 127.0.0.1 --port $PORT --device_id 0 --seed 0 \
      --test_num 10 --expert_check true --accept_expert_info_on_failure true \
      --episode_info_output $R/info_${TAG}_${t}_${suffix}.jsonl \
      --eval_batch false --additional_info eval_video_log=false > $R/log_${TAG}_${t}_${suffix}.log 2>&1
  fi
  W=$(( $(date +%s) - T0 ))
  rate=$(tr "\r" "\n" < $R/log_${TAG}_${t}_$([ "$c" = "nocache" ] && echo nc || echo $([ "$c" = "cache2" ] && echo c2 || echo c)).log | grep -a "Final success rate" | tail -1)
  ph=$(tr "\r" "\n" < $R/log_${TAG}_${t}_$([ "$c" = "nocache" ] && echo nc || echo $([ "$c" = "cache2" ] && echo c2 || echo c)).log | grep -a "phase_timing" | tail -1)
  echo -e "${TAG}\t${t}\t${c}\t${W}s\t${rate}\t${ph}" >> $EXP/summary.tsv
}
# 阶段1: 无缓存（其专家记录写入 results，随后并入 cache 供阶段2满命中）
for t in $TASKS; do run_one $t nocache; done
ln -sf $R/info_${TAG}_*_nc.jsonl $EXP/cache/ 2>/dev/null
# 阶段2: 满缓存
for t in $TASKS; do run_one $t cache; done
# 阶段3: 重复跑（可靠性）
for t in $TASKS; do run_one $t cache2; done
echo "EXP_LANE_${TAG} DONE" >> $EXP/summary.tsv
