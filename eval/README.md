# 评估工具链

所有脚本在 Radeon Cloud 实例 `/workspace/runtime/` 下实测使用，此处为最终启用的方案。
路径按官方镜像布局写死（`/RoboTwin` 代码树、`/opt/robotwin-env` Python），换环境需相应调整。

## 文件清单

| 文件 | 角色 |
|---|---|
| `run_clean_benchmark.py` | 官方 benchmark 入口（工作副本）。相对官方原版唯一改动：`--gpu-count` 的 `choices` 从 `(4, 8)` 放宽为 `(1, 4, 8)`，单卡实例必需。与 `resume_benchmark.sh` 内调用路径一致 |
| `resume_benchmark.sh` | 断点续跑入口：实例被回收/重建后一条命令接着跑（`--resume` + done 标记，已完成任务不重跑） |
| `eval_policy_xpolicylab.py.patched` | 官方评估脚本打好专家缓存补丁的成品，覆盖到 `/RoboTwin/scripts/eval_policy_xpolicylab.py` 即可用 |
| `patch_expert_cache.py` | 补丁生成器：对官方原版注入 ①缓存加载 ②专家段跳过 ③计时探针。用它对官方原版重放即可验证 `.patched` 的全部改动 |
| `open_loop_eval.py` | 官方原版 L1 开环评测脚本（随模型源码预置，未改动），位于 `/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2/scripts/`；此处收录为参考副本 |
| `open_loop_batch2.py` | 自写的批量版：复用上面官方脚本的加载/观测函数，批跑 50 任务算 MSE、hold/zero 基线与 skill 分数（`1 − MSE(model)/MSE(hold)`，消除任务间动作幅度方差）。模型只加载一次，15 分钟出全量分 |
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
- `benchmarks/official_inference.json`：官方推理 benchmark 汇总（bs=1 约 0.8–1.1s/chunk、bs=4 约 1.4s/chunk）
