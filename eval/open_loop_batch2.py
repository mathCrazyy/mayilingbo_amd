#!/usr/bin/env python3
"""L1 open-loop eval v2: model MSE + hold/zero baselines + skill score."""
import argparse, importlib.util as ilu, inspect, json, sys, time, traceback
from pathlib import Path
import numpy as np

SRC = Path("/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2")
sys.path.insert(0, str(SRC))
sys.path.insert(0, "/RoboTwin")

_spec = ilu.spec_from_file_location("ole_mod", str(SRC / "scripts" / "open_loop_eval.py"))
ole = ilu.module_from_spec(_spec)
_spec.loader.exec_module(ole)
load_policy_server = ole.load_policy_server
prepare_eval_observation = ole.prepare_eval_observation
to_numpy = ole.to_numpy
API = ole.LEROBOT_DATASET_API

from lingbotvla.data.vla_data.base_dataset import LeRobotDataset
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata


def eval_with_baselines(policy, dataset, traj_id, action_horizon, max_infer_time):
    if API == "v2":
        start_id = dataset.episode_data_index["from"][traj_id]
        end_id = dataset.episode_data_index["to"][traj_id]
    else:
        ep = dataset.meta.episodes[traj_id]
        start_id = ep["dataset_from_index"]
        end_id = ep["dataset_to_index"]
    af = policy.vla.feature_transform.org_features["actions"]
    sf = policy.vla.feature_transform.org_features["states"]
    se = 0.0; she = 0.0; sze = 0.0; n = 0
    count = 0
    for data_id in range(start_id, end_id, action_horizon):
        traj, org = prepare_eval_observation(policy, dataset[data_id])
        gt = np.concatenate([to_numpy(org[k])[:action_horizon] for k in af], axis=-1)
        st = np.concatenate([to_numpy(org[k]).reshape(1, -1) for k in sf], axis=-1)
        pred = policy.infer(traj)
        pr = np.concatenate([to_numpy(pred[k]) for k in af], axis=-1)
        if gt.shape != pr.shape:
            raise RuntimeError(f"shape mismatch gt={gt.shape} pred={pr.shape}")
        se += float(((gt - pr) ** 2).sum())
        if st.shape[1] == gt.shape[1]:
            hold = np.repeat(st, action_horizon, axis=0)
            she += float(((gt - hold) ** 2).sum())
        else:
            she = None
        sze += float((gt ** 2).sum())
        n += gt.size
        count += 1
        if count >= max_infer_time:
            break
    mse = se / n
    zero_mse = sze / n
    hold_mse = (she / n) if she is not None else None
    skill = (1 - mse / hold_mse) if hold_mse else None
    return dict(mse=mse, hold_mse=hold_mse, zero_mse=zero_mse, skill=skill)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=str(SRC.parent.parent / "models/robbyant_lingbot-vla-v2-6b"))
    p.add_argument("--norm-path", default=str(SRC / "assets/norm_stats/robotwin.json"))
    p.add_argument("--data-root", default="/RoboTwin/data/lerobot")
    p.add_argument("--traj-ids", type=int, nargs="+", default=[0, 1])
    p.add_argument("--use-length", type=int, default=25)
    p.add_argument("--max-infer-time", type=int, default=10)
    p.add_argument("--out-dir", default="/workspace/runtime/outputs/open_loop/baseline50_v2")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_jsonl = out / "results.jsonl"

    PolicyServer = load_policy_server("qwen3vl", args.model_path)
    kw = dict(path_to_pi_model=args.model_path, robot_norm_path=args.norm_path,
              use_length=args.use_length, use_bf16=True, use_fp32=False,
              chunk_ret=True, use_compile=False)
    if "video_debug_dir" in inspect.signature(PolicyServer).parameters:
        kw["video_debug_dir"] = None
    print("Loading model ...", flush=True)
    t0 = time.time()
    policy = PolicyServer(**kw)
    print(f"Model loaded in {time.time()-t0:.0f}s", flush=True)

    task_dirs = sorted(d for d in Path(args.data_root).iterdir()
                       if d.is_dir() and d.name.endswith("_joint_v30"))
    print(f"{len(task_dirs)} tasks, traj_ids={args.traj_ids}", flush=True)

    for i, ds_dir in enumerate(task_dirs):
        task = ds_dir.name.removesuffix("_joint_v30")
        t0 = time.time()
        try:
            policy.reset("robotwin")
            policy.data_config.num_episode = None
            policy.data_config.chunk_size = policy.config.chunk_size
            policy.data_config.train_path = str(ds_dir)
            policy.data_config.data_name = "robotwin"
            meta = LeRobotDatasetMetadata(ds_dir.name, root=ds_dir)
            dt = {a: [t / meta.fps for t in range(policy.config.chunk_size)]
                  for a in policy.vla.feature_transform.org_features["actions"]}
            dataset = LeRobotDataset(ds_dir.name, root=ds_dir, delta_timestamps=dt)
            recs = []
            for tid in args.traj_ids:
                r = eval_with_baselines(policy, dataset, tid, args.use_length, args.max_infer_time)
                recs.append(dict(traj=tid, **r))
                msg = {k: (round(v, 4) if v is not None else None) for k, v in r.items()}
                print(f"  traj {tid}: {msg}", flush=True)
            agg = {}
            for k in ("mse", "hold_mse", "zero_mse", "skill"):
                vs = [r[k] for r in recs if r.get(k) is not None]
                agg[k] = (sum(vs) / len(vs)) if vs else None
            rec = dict(task=task, seconds=round(time.time() - t0, 1), trajs=recs, **agg)
            status = "ok"
        except Exception as e:
            rec = dict(task=task, seconds=round(time.time() - t0, 1),
                       error=f"{type(e).__name__}: {e}")
            status = "error"
            traceback.print_exc()
        with results_jsonl.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        sec = rec.get("seconds")
        print(f"[{i+1}/{len(task_dirs)}] {task} {status} ({sec}s)", flush=True)

    rows = [json.loads(l) for l in results_jsonl.read_text().splitlines() if l.strip()]
    ok = [r for r in rows if "mse" in r]
    if ok:
        ok.sort(key=lambda r: -(r["skill"] if r.get("skill") is not None else -9))
        with (out / "summary.tsv").open("w") as f:
            f.write("task\tmse\thold_mse\tzero_mse\tskill\n")
            for r in ok:
                f.write(f"{r[chr(116)+chr(97)+chr(115)+chr(107)]}\t{r[chr(109)+chr(115)+chr(101)]:.4f}\t{r[chr(104)+chr(111)+chr(108)+chr(100)+chr(95)+chr(109)+chr(115)+chr(101)]:.4f}\t{r[chr(122)+chr(101)+chr(114)+chr(111)+chr(95)+chr(109)+chr(115)+chr(101)]:.4f}\t{r[chr(115)+chr(107)+chr(105)+chr(108)+chr(108)]:.4f}\n")
        skills = [r["skill"] for r in ok if r.get("skill") is not None]
        print(f"\nDone. {len(ok)} ok, {len(rows)-len(ok)} errors.", flush=True)
        if skills:
            print(f"Mean skill: {sum(skills)/len(skills):.4f}", flush=True)


if __name__ == "__main__":
    main()
