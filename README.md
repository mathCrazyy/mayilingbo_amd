# RoboTwin / LingBot-VLA 参赛提交：评估工具链 + 数据可视化查看器

本提交包含两个可复用部分，均在 AMD Radeon Cloud 实例（官方镜像）上实测跑通：

| 模块 | 目录 | 一句话 |
|---|---|---|
| 评估工具链 | `eval/` | 官方闭环评测的断点续跑方案、专家缓存加速（实测省 45%）、L1 开环评测批跑工具 |
| 数据可视化查看器 | `robotwin_dataviz/` | RoboTwin LeRobot v3 数据集本地/云端浏览器（纯标准库 HTTP 服务，双击即用） |

## 环境

- 硬件：AMD Radeon Cloud 单卡实例（MI 系列，48GB）
- 软件实例：官方镜像，代码树在 `/RoboTwin`，评测 Python 在 `/opt/robotwin-env/bin/python`
- 被测模型：LingBot-VLA-v2-6B（比赛官方基座，Qwen3-VL-4B + 流匹配动作专家）

## 主要实测结果

- demo_clean 50 任务 × 3 episode 全量基线：**18/150 = 12.0%**（单卡 9.7 小时）
- 专家缓存 A/B 对照（同任务同 seed）：总墙钟 **330s → 181s（省 45%）**，专家段 153.4s → 5.2s，策略段 171.7s 一秒不差
- 耗时拆解：推理仅占 5–8%，九成以上在仿真与渲染（`[phase_timing]` 探针输出）

详见各目录 README。

## 公众号

完整踩坑实录在公众号「一起富贵」连载（VLA-比赛 系列文章），欢迎关注：

<p align="left">
  <img src="assets/qrcode_pub.jpg" alt="公众号「一起富贵」二维码" width="220">
</p>
