#!/usr/bin/env python
"""RoboTwin LeRobot v3 数据集可视化服务(纯标准库 HTTP 服务,无 flask 依赖)。

用法:
    云端实例: /opt/lerobot-env/bin/python app.py            # 默认读 /RoboTwin/data/lerobot
    本地:     python app.py [数据目录] [端口]
              python app.py D:/data/lerobot 8765
本地再从云端同步数据(一次性):
    ssh radeon "cd /RoboTwin/data/lerobot && tar cf - ." | tar xf - -C <本地lerobot目录>
"""
import io
import json
import os
import re
import sys
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import av
import numpy as np
import pandas as pd
from PIL import Image

DATA_ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "/RoboTwin/data/lerobot")
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
CAMS = [
    ("observation.images.cam_high", "高位相机"),
    ("observation.images.cam_left_wrist", "左腕相机"),
    ("observation.images.cam_right_wrist", "右腕相机"),
]
CAM_SET = dict(CAMS)
JOINT_NAMES = [f"L{i}" for i in range(7)] + [f"R{i}" for i in range(7)]

_lock = threading.RLock()          # av 容器与 parquet 缓存的串行保护
_task_cache = OrderedDict()        # task -> meta dict(含全量 state/action 数组)
_task_cache_max = 6
_eps_cache = OrderedDict()         # task -> (episodes_df, fps) 轻量缓存,只读 meta
_eps_cache_max = 60
_jpeg_lru = OrderedDict()          # (task,cam,ep,frame) -> bytes
_jpeg_lru_max = 900


def _load_eps(task):
    """轻量加载:只读 episodes 元信息与 fps(不读全量数据 parquet)。"""
    with _lock:
        if task in _eps_cache:
            _eps_cache.move_to_end(task)
            return _eps_cache[task]
    root = os.path.join(DATA_ROOT, task)
    with open(os.path.join(root, "meta", "info.json")) as f:
        fps = int(json.load(f).get("fps", 15))
    eps = pd.read_parquet(os.path.join(root, "meta", "episodes", "chunk-000", "file-000.parquet"))
    val = (eps, fps)
    with _lock:
        _eps_cache[task] = val
        while len(_eps_cache) > _eps_cache_max:
            _eps_cache.popitem(last=False)
    return val


def _episode_row_of(eps, ep):
    sel = eps[eps["episode_index"] == ep]
    if sel.empty:
        raise LookupError(f"episode {ep} 不存在")
    return sel.iloc[0]


def _load_task(task):
    """加载单个任务的元信息与数据数组(带 LRU)。"""
    with _lock:
        if task in _task_cache:
            _task_cache.move_to_end(task)
            return _task_cache[task]
    root = os.path.join(DATA_ROOT, task)
    with open(os.path.join(root, "meta", "info.json")) as f:
        info = json.load(f)
    eps = pd.read_parquet(os.path.join(root, "meta", "episodes", "chunk-000", "file-000.parquet"))
    tasks_tbl = pd.read_parquet(os.path.join(root, "meta", "tasks.parquet"))
    if "task" in tasks_tbl.columns:
        task_texts = tasks_tbl["task"].tolist()
    else:  # 文本在索引上,task_index 列给出编号
        task_texts = [str(v) for v in tasks_tbl.sort_values("task_index").index]
    df = pd.read_parquet(os.path.join(root, "data", "chunk-000", "file-000.parquet"))
    state = np.vstack([np.asarray(v, dtype=np.float32) for v in df["observation.state"]])
    action = np.vstack([np.asarray(v, dtype=np.float32) for v in df["action"]])
    meta = {
        "root": root, "eps": eps, "task_texts": task_texts,
        "fps": int(info.get("fps", 15)),
        "n_rows": len(df),
        "ep_index": df["episode_index"].to_numpy(),
        "ts": df["timestamp"].to_numpy(dtype=np.float64),
        "task_index": df["task_index"].to_numpy(),
        "state": state, "action": action,
    }
    with _lock:
        _task_cache[task] = meta
        while len(_task_cache) > _task_cache_max:
            _task_cache.popitem(last=False)
    return meta


def _list_tasks():
    out = []
    for name in sorted(os.listdir(DATA_ROOT)):
        if os.path.exists(os.path.join(DATA_ROOT, name, "meta", "info.json")):
            out.append(name)
    if not out:
        raise RuntimeError(f"未在 {DATA_ROOT} 找到 LeRobot 数据集")
    return out


def _episode_row(meta, ep):
    sel = meta["eps"][meta["eps"]["episode_index"] == ep]
    if sel.empty:
        raise LookupError(f"episode {ep} 不存在")
    return sel.iloc[0]


def _decode_jpeg(task, cam, ep, ep_row, frame, fps):
    key = (task, cam, int(ep), int(frame))
    with _lock:
        if key in _jpeg_lru:
            _jpeg_lru.move_to_end(key)
            return _jpeg_lru[key]
    chunk = int(ep_row[f"videos/{cam}/chunk_index"])
    f_idx = int(ep_row[f"videos/{cam}/file_index"])
    vpath = os.path.join(DATA_ROOT, task, "videos", cam,
                         f"chunk-{chunk:03d}", f"file-{f_idx:03d}.mp4")
    from_ts = float(ep_row[f"videos/{cam}/from_timestamp"])
    to_ts = float(ep_row[f"videos/{cam}/to_timestamp"])
    target = min(from_ts + frame / fps, max(from_ts, to_ts - 0.5 / fps))
    with _lock:
        container = av.open(vpath)
        try:
            stream = container.streams.video[0]
            container.seek(int(max(0.0, target - 0.25) * 1_000_000))
            tol = 0.5 / fps
            img = None
            for pic in container.decode(stream):
                if pic.time is None:
                    continue
                if pic.time >= target - tol:
                    img = pic.to_image()
                    break
            if img is None:
                img = Image.new("RGB", (320, 240), (20, 20, 20))
        finally:
            container.close()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    data = buf.getvalue()
    with _lock:
        _jpeg_lru[key] = data
        while len(_jpeg_lru) > _jpeg_lru_max:
            _jpeg_lru.popitem(last=False)
    return data


# ---------------------------------------------------------------- API 数据
def api_tasks():
    out = []
    for name in _list_tasks():
        try:
            meta = _load_task(name)
        except Exception as e:
            out.append({"name": name, "error": str(e)})
            continue
        out.append({
            "name": name,
            "episodes": int(len(meta["eps"])),
            "frames": int(meta["n_rows"]),
            "seconds": round(float(meta["ts"].max()), 1) if len(meta["ts"]) else 0,
            "fps": meta["fps"],
            "avg_len": round(float(meta["eps"]["length"].mean()), 1),
        })
    return out


def _first_text(val):
    """episodes 表 tasks 列是字符串数组,取第一条;空则回退 None。"""
    try:
        if val is None:
            return None
        if hasattr(val, "__len__") and len(val):
            return str(val[0])
    except Exception:
        pass
    return None


def api_episodes(task):
    """轻量:每集帧数与指令,不读全量数据。"""
    eps, _ = _load_eps(task)
    out = []
    for _, r in eps.sort_values("episode_index").iterrows():
        out.append({"ep": int(r["episode_index"]), "len": int(r["length"]),
                    "text": _first_text(r.get("tasks")) or ""})
    return out


def api_episode(task, ep):
    meta = _load_task(task)
    row = _episode_row(meta, ep)
    sel = np.flatnonzero(meta["ep_index"] == ep)
    if sel.size == 0:
        raise LookupError("episode 无数据行")
    text = _first_text(row.get("tasks"))
    if not text:
        ti = int(meta["task_index"][sel[0]])
        text = meta["task_texts"][ti] if ti < len(meta["task_texts"]) else str(ti)
    cams = {cam: {"from": float(row[f"videos/{cam}/from_timestamp"]),
                  "to": float(row[f"videos/{cam}/to_timestamp"])} for cam, _ in CAMS}
    return {
        "task": task, "episode": int(ep), "n_episodes": int(len(meta["eps"])),
        "instruction": text, "fps": meta["fps"], "frames": int(sel.size),
        "ts": [round(float(t), 4) for t in meta["ts"][sel]],
        "state": np.round(meta["state"][sel], 5).tolist(),
        "action": np.round(meta["action"][sel], 5).tolist(),
        "cams": cams,
    }


# ---------------------------------------------------------------- HTML
BASE_CSS = """
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--fg:#d7dce4;--dim:#8b93a3;--ac:#4f9cf9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,"Segoe UI",Roboto,"Microsoft YaHei",sans-serif}
a{color:var(--ac);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1280px;margin:0 auto;padding:16px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}
h1{font-size:20px;margin:0 0 4px}.muted{color:var(--dim);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.card b{display:block;margin-bottom:6px}
table{border-collapse:collapse;width:100%}th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--dim);font-weight:500;font-size:12px}
tr:hover td{background:#1c2029}
input[type=range]{width:100%;accent-color:var(--ac)}
button{background:#222734;color:var(--fg);border:1px solid var(--line);border-radius:6px;
padding:6px 14px;cursor:pointer}button:hover{border-color:var(--ac)}
button.primary{background:var(--ac);border-color:var(--ac);color:#fff}
.chip{display:inline-block;background:#222734;border:1px solid var(--line);border-radius:20px;
padding:2px 10px;margin:2px;cursor:pointer;font-size:12px;user-select:none}
.chip.on{background:var(--ac);color:#fff;border-color:var(--ac)}
"""

HOME = """<!doctype html><html><head><meta charset=utf-8><title>RoboTwin 数据集浏览</title>
<style>__CSS__</style></head><body><div class=wrap>
<h1>RoboTwin 2.0 · LeRobot v3 数据集</h1>
<div class=muted>__ROOT__ · 共 <span id=n></span> 个任务 ·
<a href="/compare">视觉相似度分析 →</a></div>
<div class=grid id=grid></div></div>
<script>
fetch('/api/tasks').then(r=>r.json()).then(ts=>{
document.getElementById('n').textContent=ts.length;
const g=document.getElementById('grid');
ts.sort((a,b)=>a.name.localeCompare(b.name)).forEach(t=>{
 const d=document.createElement('div');d.className='card';
 if(t.error){d.innerHTML=`<b>${t.name}</b><span class=muted>加载失败:${t.error}</span>`;}
 else{d.innerHTML=`<a href="/task/${t.name}"><b>${t.name}</b></a>
 <span class=muted>${t.episodes} episodes · ${t.frames} 帧 · 平均 ${(t.avg_len/t.fps).toFixed(1)}s/集(${t.avg_len} 帧)@${t.fps}fps</span>`;}
 g.appendChild(d);});
});</script></body></html>"""

TASK_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>__TASK__</title>
<style>__CSS__</style></head><body><div class=wrap>
<a href="/">← 任务列表</a>
<h1>__TASK__</h1><div class=muted></div>
<div class=panel><table><thead><tr><th>Episode</th><th>指令</th><th>帧数</th><th>时长</th><th></th>
</tr></thead><tbody id=tb></tbody></table></div></div>
<script>
const TASK='__TASK__';
fetch('/api/tasks').then(r=>r.json()).then(all=>{
 const t=all.find(x=>x.name===TASK);
 if(t)document.querySelector('.muted').textContent=
  `${t.episodes} episodes · ${t.frames} 帧 · 平均 ${(t.avg_len/t.fps).toFixed(1)}s/集 @ ${t.fps} fps`;
 const eps=__EPS__;
 const tb=document.getElementById('tb');
 eps.forEach(e=>{
  const tr=document.createElement('tr');
  tr.innerHTML=`<td>${e.ep}</td><td>${e.text}</td><td>${e.len}</td>
   <td>${(e.len/15).toFixed(1)}s</td>
   <td><a href="/episode/${TASK}/${e.ep}">查看 →</a></td>`;
  tb.appendChild(tr);});
});</script></body></html>"""

EPISODE_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>__TASK__ #__EP__</title>
<style>__CSS__
.cams{display:grid;grid-template-columns:2fr 1fr 1fr;gap:8px}
.cams figure{margin:0;background:#000;border-radius:8px;overflow:hidden;position:relative}
.cams canvas{width:100%;height:auto;display:block}
.cams figcaption{position:absolute;left:6px;top:4px;background:#0009;font-size:11px;
padding:1px 8px;border-radius:10px}
.layout{display:grid;grid-template-columns:480px 1fr;gap:14px}
.ctrl{display:flex;gap:8px;align-items:center;margin:10px 0;flex-wrap:wrap}
#pos{font-variant-numeric:tabular-nums}
.chartbox{height:240px;margin-bottom:10px}
@media(max-width:1100px){.layout{grid-template-columns:1fr}}
</style></head><body><div class=wrap>
<div><a href="/task/__TASK__">← __TASK__</a>
 &nbsp;<a id=prevEp href="#">← 上一 episode</a>
 &nbsp;<a id=nextEp href="#">下一 episode →</a></div>
<h1>__TASK__ · episode #__EP__</h1>
<div class=muted id=meta></div>
<div class=panel id=instr style=font-size:15px></div>
<div class=layout>
 <div>
  <div class=cams>
   <figure><canvas id=cv_high width=320 height=240></canvas><figcaption>高位相机</figcaption></figure>
   <figure><canvas id=cv_left width=320 height=240></canvas><figcaption>左腕</figcaption></figure>
   <figure><canvas id=cv_right width=320 height=240></canvas><figcaption>右腕</figcaption></figure>
  </div>
  <div class=ctrl>
   <button id=bFirst>⏮</button><button id=bPrev>◀</button>
   <button id=bPlay class=primary>▶ 播放</button>
   <button id=bNext>▶</button><button id=bLast>⏭</button>
   <select id=speed style="background:#222734;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:5px">
    <option value=0.25>0.25×</option><option value=0.5>0.5×</option>
    <option value=1 selected>1×</option><option value=2>2×</option><option value=4>4×</option>
   </select>
   <span id=pos></span>
   <span class=muted id=videoLinks></span>
  </div>
  <input type=range id=slider min=0 value=0 step=1>
  <div style=margin-top:6px>
   <span class=muted>关节:</span><span id=chips></span>
  </div>
 </div>
 <div>
  <div class=chartbox><canvas id=cState></canvas></div>
  <div class=chartbox><canvas id=cAction></canvas></div>
 </div>
</div></div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<script>
const TASK='__TASK__',EP=__EP__;
const CAMS=[['observation.images.cam_high','cv_high'],['observation.images.cam_left_wrist','cv_left'],['observation.images.cam_right_wrist','cv_right']];
const JN=__JN__;const NC=JN.length;
const COLORS=['#e6194b','#f58231','#ffd700','#bfef45','#3cb44b','#42d4f4','#4363d8',
              '#911eb4','#f032e6','#fabed4','#469990','#dcbeff','#9a6324','#800000'];
let D=null,frame=0,playing=false,timer=null;
const $=id=>document.getElementById(id);

function setFrameURLs(){
 const f=frame;
 CAMS.forEach(([c,id])=>{
  const im=new Image();
  im.onload=()=>{ if(f!==frame) return;   // 帧令牌:乱序返回的旧帧丢弃
   const cv=document.getElementById(id);
   const cx=cv.getContext('2d');
   cv.width=320; cv.height=240;           // 重设尺寸即清空,再整帧绘制
   cx.drawImage(im,0,0,320,240); };
  im.src=`/frame/${TASK}/${EP}/${c}/${f}.jpg`;
 });
 if(D) for(let k=1;k<=2&&frame+k<D.frames;k++)
  CAMS.forEach(([c])=>{const im=new Image();im.src=`/frame/${TASK}/${EP}/${c}/${frame+k}.jpg`;});
}
function syncUI(){
 $('slider').max=D.frames-1;$('slider').value=frame;
 $('pos').textContent=`帧 ${frame+1} / ${D.frames} · ${D.ts[frame].toFixed(2)}s`;
 Object.values(CH).forEach(ch=>ch&&ch.update('none'));
}
function go(f){if(!D)return;frame=Math.max(0,Math.min(D.frames-1,f));setFrameURLs();syncUI();}

const cursorPlugin={id:'cursor',frame:0,
 afterDatasetsDraw(chart){const t=this.frame/D.fps;
  const x=chart.scales.x.getPixelForValue(t);
  if(!isFinite(x))return;
  const{top,bottom}=chart.chartArea;const ctx=chart.ctx;
  ctx.save();ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.setLineDash([4,3]);
  ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom);ctx.stroke();ctx.restore();}};

let CH={};
function makeChart(cid,key,title){
 const ctx=$(cid).getContext('2d');
 const datasets=JN.map((n,j)=>({label:n,
  data:D.ts.map((t,i)=>({x:t,y:D[key][i][j]})),
  borderColor:COLORS[j%COLORS.length],borderWidth:1.3,pointRadius:0,tension:0}));
 const ch=new Chart(ctx,{type:'line',
  data:{datasets},
  options:{responsive:true,maintainAspectRatio:false,animation:false,
   interaction:{mode:'nearest',intersect:false},
   parsing:false,
   plugins:{title:{display:true,text:title,color:'#8b93a3'},
    legend:{display:false}},
   scales:{x:{type:'linear',title:{display:true,text:'时间 (s)'},ticks:{color:'#8b93a3',maxTicksLimit:10},
     grid:{color:'#222734'}},
    y:{ticks:{color:'#8b93a3'},grid:{color:'#222734'}}}},
  plugins:[cursorPlugin]});
 ch.canvas.onclick=e=>{const r=ch.canvas.getBoundingClientRect();
  const xV=ch.scales.x.getValueForPixel(e.clientX-r.left);
  if(xV!==undefined&&xV!==null)go(Math.round(xV*D.fps));};
 return ch;
}
let showJoint=Array(NC).fill(true);
function applyChips(){
 Object.values(CH).forEach(ch=>ch.data.datasets.forEach((ds,j)=>ds.hidden=!showJoint[j]));
 Object.values(CH).forEach(ch=>ch.update('none'));refreshChips();
}
function refreshChips(){const c=$('chips');c.innerHTML='';
 [['左臂',i=>i<7],['右臂',i=>i>=7],['全部',null]].forEach(([label,filt])=>{
  const on=filt?showJoint.filter((v,i)=>filt(i)).every(Boolean):showJoint.every(Boolean);
  const s=document.createElement('span');s.className='chip'+(on?' on':'');s.textContent=label;
  s.onclick=()=>{const target=!on;
   for(let i=0;i<NC;i++)if(!filt||filt(i))showJoint[i]=target;applyChips();};
  c.appendChild(s);});
}
refreshChips();

function pause(){playing=false;clearInterval(timer);$('bPlay').textContent='▶ 播放';}
function play(){if(playing){pause();return;}
 playing=true;$('bPlay').textContent='⏸ 暂停';
 const speed=parseFloat($('speed').value);
 timer=setInterval(()=>{if(frame>=D.frames-1){pause();return;}go(frame+1);},1000/(D.fps*speed));}

fetch(`/api/episode/${TASK}/${EP}`).then(r=>r.json()).then(d=>{
 D=d;$('instr').textContent='📋 '+d.instruction;
 $('meta').textContent=`共 ${d.n_episodes} episodes · 本集 ${d.frames} 帧 @ ${d.fps} fps`;
 if(EP>0)$('prevEp').href=`/episode/${TASK}/${EP-1}`;
 if(EP<d.n_episodes-1){$('nextEp').href=`/episode/${TASK}/${EP+1}`;}
 else $('nextEp').style.display='none';
 $('videoLinks').innerHTML='流畅视频: '+CAMS.map(([c],i)=>
  `<a href="/video/${TASK}/${c}/0#t=${d.cams[c].from},${d.cams[c].to}" target=_blank>${['高位','左腕','右腕'][i]}</a>`).join(' · ');
 CH.state=makeChart('cState','state','observation.state(当前关节位置,rad)');
 CH.action=makeChart('cAction','action','action(目标关节位置,rad)');
 go(0);});

$('bPlay').onclick=play;
$('bFirst').onclick=()=>go(0);$('bLast').onclick=()=>{if(D)go(D.frames-1);};
$('bPrev').onclick=()=>go(frame-1);$('bNext').onclick=()=>go(frame+1);
$('slider').oninput=e=>{pause();go(+e.target.value);};
$('speed').onchange=()=>{if(playing){pause();play();}};
document.addEventListener('keydown',e=>{
 if(e.key===' '){e.preventDefault();play();}
 if(e.key==='ArrowLeft')go(frame-1);if(e.key==='ArrowRight')go(frame+1);});
</script></body></html>"""


COMPARE_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>视觉相似度分析</title>
<style>__CSS__
.ctrl{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
.ctrl select,.ctrl input{background:#222734;color:var(--fg);border:1px solid var(--line);
border-radius:6px;padding:6px 8px}
.ctrl input[type=number]{width:90px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:10px}
.cell{background:var(--panel);border:2px solid var(--line);border-radius:8px;overflow:hidden}
.cell canvas{width:100%;display:block}
.cell .cap{padding:4px 8px;font-size:12px;display:flex;justify-content:space-between;
gap:6px;align-items:baseline}
.cell .cap .lb{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.badge{font-variant-numeric:tabular-nums;font-weight:600;flex-shrink:0}
.sim{border-color:#3cb44b}.mid{border-color:#ffd700}.dif{border-color:#e6194b}
.hint{margin:4px 0 14px}
</style></head><body><div class=wrap>
<h1>高位相机 · 指定帧相似度分析</h1>
<div class=ctrl>
 <select id=task></select>
 <span class=muted>帧号</span><input type=number id=frame value=0 min=0>
 <button id=bM>-10</button><button id=bP>+10</button><button id=bMid>中间帧</button>
 <button id=bReload class=primary>刷新</button>
 <a href="/" style="margin-left:auto">← 返回</a>
</div>
<div class="muted hint">差值 = 该格与第 1 格在 80×60 灰度下的平均像素差(0–255)。
绿 ≤6(几乎相同)· 黄 ≤18(相近)· 红 >18(差异明显)。帧号超出该集长度时取末帧。</div>
<div class=muted id=summary style=margin-bottom:10px></div>
<div class=grid id=grid></div></div>
<script>
const P=new URLSearchParams(location.search);
let TASK=P.get('task')||'',FRAME=Math.max(0,parseInt(P.get('frame')||'0')||0);
const $=id=>document.getElementById(id);
const OFF=document.createElement('canvas');OFF.width=80;OFF.height=60;
const OX=OFF.getContext('2d',{willReadFrequently:true});

function gray(img){OX.drawImage(img,0,0,80,60);
 const d=OX.getImageData(0,0,80,60).data,g=new Float32Array(80*60);
 for(let i=0,p=0;i<g.length;i++,p+=4)g[i]=(d[p]+d[p+1]+d[p+2])/3;
 return g;}
function diff(a,b){let s=0;for(let i=0;i<a.length;i++)s+=Math.abs(a[i]-b[i]);return s/a.length;}

async function initSelect(){
 const ts=await(await fetch('/api/tasks')).json();
 $('task').innerHTML='<option value="__all__">—— 全部任务(各取 episode 0)——</option>'+
  ts.map(t=>`<option value="${t.name}">${t.name.replace('_joint_v30','')}</option>`).join('');
 if(!TASK)TASK=ts[0].name;
 $('task').value=TASK;
 $('task').onchange=()=>{TASK=$('task').value;render();};
}
async function buildItems(){
 if(TASK==='__all__'){
  const ts=await(await fetch('/api/tasks')).json();
  const eps=await Promise.all(ts.map(t=>fetch('/api/episodes/'+t.name).then(r=>r.json())));
  return ts.map((t,i)=>({task:t.name,ep:0,len:eps[i][0].len,
   label:t.name.replace('_joint_v30','')}));
 }
 const eps=await(await fetch('/api/episodes/'+TASK)).json();
 return eps.map(e=>({task:TASK,ep:e.ep,len:e.len,label:'ep '+e.ep}));
}
async function render(){
 history.replaceState(null,'',`/compare?task=${encodeURIComponent(TASK)}&frame=${FRAME}`);
 $('frame').value=FRAME;
 const items=await buildItems();
 $('bMid').onclick=()=>{FRAME=Math.round(items[0].len/2);render();};
 const grid=$('grid');grid.innerHTML='';
 const cells=items.map(it=>{
  const f=Math.min(FRAME,it.len-1);
  const cell=document.createElement('div');cell.className='cell';
  cell.innerHTML=`<canvas width=320 height=240></canvas>
   <div class=cap><span class=lb>${it.label}${f<FRAME?' <span class=muted>末帧</span>':''}</span>
   <span class=badge>…</span></div>`;
  grid.appendChild(cell);
  return {it,f,cv:cell.querySelector('canvas'),bd:cell.querySelector('.badge'),cell};
 });
 const grays=await Promise.all(cells.map(({it,f,cv})=>new Promise(res=>{
  const im=new Image();
  im.onload=()=>{const cx=cv.getContext('2d');cx.drawImage(im,0,0,320,240);res(gray(im));};
  im.onerror=()=>res(null);
  im.src=`/frame/${it.task}/${it.ep}/observation.images.cam_high/${f}.jpg`;
 })));
 const ref=grays.find(g=>g);
 const ds=grays.map(g=>g&&ref?diff(g,ref):null);
 ds.forEach((d,i)=>{
  const c=cells[i];
  if(d===null){c.bd.textContent='失败';c.cell.classList.add('dif');return;}
  c.bd.textContent=d.toFixed(1);
  c.cell.classList.add(d<=6?'sim':d<=18?'mid':'dif');
 });
 const v=ds.filter(d=>d!==null&&ds.indexOf(d)>0);
 const rest=ds.slice(1).filter(d=>d!==null);
 const mean=rest.reduce((a,b)=>a+b,0)/(rest.length||1);
 const sorted=[...rest].sort((a,b)=>a-b);
 const med=sorted.length?sorted[sorted.length>>1]:0;
 $('summary').innerHTML=`<b>${items.length}</b> 格 · 帧号 ${FRAME} ·
  与第 1 格差值:平均 <b>${mean.toFixed(1)}</b> · 中位 <b>${med.toFixed(1)}</b> ·
  最大 <b>${sorted.length?sorted[sorted.length-1].toFixed(1):'-'}</b> ·
  几乎相同(≤6):<b>${rest.filter(d=>d<=6).length}</b> / ${rest.length}`;
}
initSelect().then(()=>render());
$('bP').onclick=()=>{FRAME+=10;render();};
$('bM').onclick=()=>{FRAME=Math.max(0,FRAME-10);render();};
$('bReload').onclick=()=>render();
$('frame').onchange=()=>{FRAME=Math.max(0,parseInt($('frame').value)||0);render();};
</script></body></html>"""


def page_home():
    return HOME.replace("__CSS__", BASE_CSS).replace("__ROOT__", DATA_ROOT)


def page_task(task):
    meta = _load_task(task)
    eps = meta["eps"].sort_values("episode_index")
    rows = []
    for _, r in eps.iterrows():
        text = _first_text(r.get("tasks"))
        if not text:
            sel0 = np.flatnonzero(meta["ep_index"] == int(r["episode_index"]))
            ti = int(meta["task_index"][sel0[0]]) if sel0.size else 0
            text = meta["task_texts"][ti] if ti < len(meta["task_texts"]) else ""
        rows.append({"ep": int(r["episode_index"]), "len": int(r["length"]), "text": text})
    return (TASK_PAGE.replace("__CSS__", BASE_CSS).replace("__TASK__", task)
            .replace("__EPS__", json.dumps(rows, ensure_ascii=False)))


def page_episode(task, ep):
    _load_task(task)  # 校验任务存在
    return (EPISODE_PAGE.replace("__CSS__", BASE_CSS).replace("__TASK__", task)
            .replace("__EP__", str(ep)).replace("__JN__", json.dumps(JOINT_NAMES)))


# ---------------------------------------------------------------- HTTP 层
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RoboTwinViz/1.0"

    def log_message(self, fmt, *args):
        pass  # 静默访问日志

    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _html(self, text):
        self._send(200, text.encode("utf-8"), "text/html; charset=utf-8")

    def _error(self, code, msg):
        self._json({"error": msg}, code)

    def _serve_mp4(self, vpath):
        size = os.path.getsize(vpath)
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng.strip())
            start = int(m.group(1)) if m and m.group(1) else 0
            end = int(m.group(2)) if m and m.group(2) else size - 1
            end = min(end, size - 1)
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(vpath, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(vpath, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/" or path == "/index.html":
                return self._html(page_home())
            m = re.fullmatch(r"/api/tasks", path)
            if m:
                return self._json(api_tasks())
            m = re.fullmatch(r"/api/episodes/([^/]+)", path)
            if m:
                return self._json(api_episodes(m.group(1)))
            m = re.fullmatch(r"/api/episode/([^/]+)/(\d+)", path)
            if m:
                return self._json(api_episode(m.group(1), int(m.group(2))))
            if path == "/compare":
                return self._send(200, COMPARE_PAGE.replace("__CSS__", BASE_CSS).encode("utf-8"),
                                  "text/html; charset=utf-8")
            m = re.fullmatch(r"/frame/([^/]+)/(\d+)/([^/]+)/(\d+)\.jpg", path)
            if m:
                task, ep, cam, frame = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
                if cam not in CAM_SET:
                    return self._error(400, "bad camera")
                eps, fps = _load_eps(task)
                row = _episode_row_of(eps, ep)
                jpg = _decode_jpeg(task, cam, ep, row, frame, fps)
                return self._send(200, jpg, "image/jpeg",
                                  {"Cache-Control": "public, max-age=3600"})
            m = re.fullmatch(r"/video/([^/]+)/([^/]+)/(\d+)", path)
            if m:
                task, cam, f_idx = m.group(1), m.group(2), int(m.group(3))
                if cam not in CAM_SET:
                    return self._error(400, "bad camera")
                vpath = os.path.join(DATA_ROOT, task, "videos", cam,
                                     "chunk-000", f"file-{f_idx:03d}.mp4")
                if not os.path.exists(vpath):
                    return self._error(404, "video not found")
                return self._serve_mp4(vpath)
            m = re.fullmatch(r"/task/([^/]+)", path)
            if m:
                return self._html(page_task(m.group(1)))
            m = re.fullmatch(r"/episode/([^/]+)/(\d+)", path)
            if m:
                return self._html(page_episode(m.group(1), int(m.group(2))))
            return self._error(404, f"not found: {path}")
        except LookupError as e:
            return self._error(404, str(e))
        except FileNotFoundError as e:
            return self._error(404, str(e))
        except Exception as e:
            return self._error(500, f"{type(e).__name__}: {e}")


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"RoboTwin 数据可视化服务已启动: http://127.0.0.1:{PORT}  (数据目录 {DATA_ROOT})")
    server.serve_forever()


if __name__ == "__main__":
    main()
