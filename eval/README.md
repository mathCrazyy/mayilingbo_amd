# 评估工具链

所有脚本在 Radeon Cloud 实例 `/workspace/runtime/` 下实测使用，此处为最终启用的方案。
路径按官方镜像布局写死（`/RoboTwin` 代码树、`/opt/robotwin-env` Python），换环境需相应调整。

## 文件清单

| 文件 | 角色 |
|---|---|
| `run_clean_benchmark.py` | 官方 benchmark 入口（工作副本）。相对官方原版唯一改动：`--gpu-count` 的 `choices` 从 `(4, 8)` 放宽为 `(1, 4, 8)`，单卡实例必需。与 `resume_benchmark.sh` 内调用路径一致 |
| `resume_benchmark.sh` | 断点续跑入口：实例被回收/重建后一条命令接着跑（`--resume` + done 标记，已完成任务不重跑） |
| `eval_policy_xpolicylab.py.patched` | 官方评估脚本的完整补丁成品，覆盖到 `/RoboTwin/scripts/eval_policy_xpolicylab.py` 即可用。包含 ①专家缓存（支持目录模式）②缓存命中时从 `info["{a}"]` 恢复 `arm_tag`（否则 open_laptop 等任务 check_success 秒崩）③逐集按 crc32(任务+seed) 派生种子并随 reset 消息传给服务端 ④指令措辞固定取第一条（消除随机挑话术）⑤`[phase_timing]` 计时探针 |
| `lingbot_vla_v2_policy.py.patched_determ` | 服务端（模型推理服务）确定性补丁：reset 消息带 `reset_seed` 时 `set_seed_everywhere()` 重设随机数，切断流匹配噪声按调用次序发放的级联。与上面客户端 ③④ 配套，覆盖到 `deploy/lingbot_vla_v2_policy.py` |
| `patch_expert_cache.py` | 补丁生成器：对官方原版注入 ①缓存加载 ②专家段跳过 ③计时探针。用它对官方原版重放即可验证 `.patched` 的全部改动 |
| `open_loop_eval.py` | 官方原版 L1 开环评测脚本（随模型源码预置，未改动），位于 `/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2/scripts/`；此处收录为参考副本 |
| `open_loop_batch2.py` | 自写的批量版：复用上面官方脚本的加载/观测函数，批跑 50 任务算 MSE、hold/zero 基线与 skill 分数（`1 − MSE(model)/MSE(hold)`，消除任务间动作幅度方差）。模型只加载一次，15 分钟出全量分 |
| `exp_lane.sh` | 缓存 2×2×2 对照实验驱动：{基座, 训练版} × {无缓存, 满缓存, 重复} × 5 代表任务 × 10 episode（30 次评测）。用法 `exp_lane.sh <PORT> <TAG>`，基座/训练版各起一个服务端 lane |
| `eps_scaling.sh` | 单任务 episode 递增实验：N=10/20/30/50/100 依次跑（训练版+满缓存），验证耗时线性与成功率收敛 |
| `stapler_seedcheck.sh` | press_stapler 灾难性遗忘多 seed 复测：`--seed_start` 200000/300000/400000 三段 × 10 episode，基座 vs 训练版各一遍 |
| `determ_aa_test.sh` | 确定性 A/A 测试：同一任务同 seed 连跑两遍，对比逐集结果是否一字不差 |
| `outputs/` | 数据与证据（见下） |

## 用法

### 1. 闭环评测（官方口径）

```bash
# 先起官方模型服务（13400 端口），再：
bash resume_benchmark.sh          # = run_clean_benchmark.py --gpu-count 1 --task-config demo_clean --episodes 3 --resume
```

### 2. 专家缓存加速（复评立省）

评测每个场景前会让脚本专家先试玩一遍（验证可解），与被测策略无关且 seed 固定 → 可缓存：

```bash
# a. 用历史评测的 episode_info 合并出缓存（每行一条 (task_config, task_name, seed) 记录）
cat outputs/demo_clean_50x3_1gpu/episode_info/*.jsonl > expert_cache.jsonl

# b. 打补丁（或直接用 eval_policy_xpolicylab.py.patched 覆盖）
python patch_expert_cache.py

# c. 挂缓存重跑
ROBOTWIN_EXPERT_CACHE=expert_cache.jsonl bash resume_benchmark.sh
```

命中时日志打 `[expert_cache] HIT seed=...`，未命中回退现场重跑专家段；每个任务结束打
`[phase_timing] expert_total=...s policy_total=...s` 用于耗时拆解。

**实测**（同任务同 3 episode 同模型，`outputs/expert_cache_test/`）：总墙钟 330s→181s，专家段 153.4s→5.2s，策略段 171.7s 不变。

### 3. 开环评测（L1 初筛仪表）

`open_loop_batch2.py` 按 `/RoboTwin/.../lingbot-vla-v2/scripts/open_loop_eval.py` 的绝对路径 import 官方脚本（实例上已预置，仓库里附了同文件副本供参考）：

```bash
python open_loop_batch2.py --out-dir outputs/open_loop/baseline50_v2
```

结论性提醒：开环 MSE/skill 与闭环成功率不相关（有分任务均值 MSE 0.694 vs 全零任务 0.648；50 任务 skill 全负，均值 −11.8），只作训练前后对比的仪表用，评测定论以闭环为准。

## outputs/ 说明

- `demo_clean_50x3_1gpu/episode_info/`（50 文件）：50×3 全量基线（总分 18/150 = 12.0%，单卡 9.7 小时）的专家段记录，也是专家缓存的数据源（133/150 条覆盖）
- `expert_cache_test/`：缓存 A/B 对照日志（runA 无缓存 / runB 挂缓存）
- `exp_2x2/summary.tsv`：30 次评测对照实验全量结果（模型 × 缓存条件 × 任务 × 墙钟 × 成功率 × 耗时拆解）。核心结论：缓存省时 基座 36% / 训练版 67%；30 次中 27 次成功率完全一致，仅 2 个边缘 episode 翻转（位级随机，非缓存偏差）；训练版把 press_stapler 从满分练到三成（灾难性遗忘）
- `eps_scaling/summary.txt`：episode 递增实验（N=10→100 墙钟 88→1015s 严格线性，约 10s/集；成功率 50%→72%）
- `stapler_seedcheck/summary.txt`：press_stapler 多 seed 段复测——基座 10/10 × 6（含标准段三次），训练版新三段 7/10、2/10、1/10，遗忘坐实
- `benchmarks/official_inference.json`：官方推理 benchmark 汇总（bs=1 约 0.8–1.1s/chunk、bs=4 约 1.4s/chunk）
