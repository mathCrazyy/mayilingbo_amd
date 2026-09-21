# RoboTwin LeRobot v3 数据集可视化查看器

纯 Python 标准库 HTTP 服务（无 Flask 等框架依赖），浏览器里按任务/episode/帧浏览 RoboTwin 的
LeRobot v3 数据集：三路相机画面（高位/左腕/右腕）、14 维关节曲线、语言指令，逐帧对齐回放。

## 依赖

`av`（解码 mp4 视频）、`numpy`、`pandas`（读 parquet）、`pillow`。见 `requirements.txt`。

## 启动

**本地（Windows）**

```bat
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python app.py data\lerobot 8765
```

或双击 `启动查看器.bat`（自动开浏览器）。

**云端实例（Linux）**

```bash
/opt/robotwin-env/bin/python app.py            # 默认读 /RoboTwin/data/lerobot，端口 8765
```

## 数据获取

本地数据从云端实例一次性同步（网络受限环境用 tar over ssh）：

```bash
mkdir -p data/lerobot
ssh radeon "cd /RoboTwin/data/lerobot && tar cf - ." | tar xf - -C data/lerobot
```

## 说明

- 视频解码与 parquet 读取带 LRU 缓存（任务级 6 个、episode 元信息 60 个），50 任务数据下翻页不卡
- URL 支持 `/task/<任务名>`、`/episode/<任务名>/<集号>` 直达，另有 `/compare` 任务对比页，方便在文档/报告里引用具体位置
