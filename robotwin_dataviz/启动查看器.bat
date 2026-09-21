@echo off
rem RoboTwin 数据集可视化查看器 - 本地启动
cd /d "%~dp0"
start "" http://127.0.0.1:8765/
venv\Scripts\python.exe app.py data\lerobot 8765
