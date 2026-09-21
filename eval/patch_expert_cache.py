import ast

P = "/RoboTwin/scripts/eval_policy_xpolicylab.py"
s = open(P, encoding="utf-8").read()

def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, f"marker count {c} != {n}: {old[:60]!r}"
    s = s.replace(old, new)

# 1) helper: cache loader + accumulators, inserted before build_instruction
helper = """import os as _os
import json as _json

_EXPERT_CACHE = {}
_EXPERT_CACHE_PATH = _os.environ.get("ROBOTWIN_EXPERT_CACHE")
if _EXPERT_CACHE_PATH:
    with open(_EXPERT_CACHE_PATH, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line:
                continue
            try:
                _r = _json.loads(_line)
            except Exception:
                continue
            _EXPERT_CACHE[(_r.get("task_config"), _r.get("task_name"), int(_r.get("seed")))] = _r
    print(f"[expert_cache] loaded {len(_EXPERT_CACHE)} records from {_EXPERT_CACHE_PATH}")

_ACC = {"expert": 0.0, "policy": 0.0}


def _expert_cache_get(task_config, task_name, seed):
    return _EXPERT_CACHE.get((task_config, task_name, int(seed)))


def build_instruction("""
rep("def build_instruction(", helper)

# 2) wrap expert sim block with cache guard (re-indent original by +4)
S_START = "            try:\n                task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n                episode_info = task_env.play_once()"
S_END = "                    print(f\"error occurs during expert check! seed={now_seed - 1} err={type(e).__name__}: {e}\")\n                    continue"
i0 = s.find(S_START)
i1 = s.find(S_END)
assert i0 != -1 and i1 != -1 and i1 > i0, "expert region markers not found"
i1_end = i1 + len(S_END)
region = s[i0:i1_end]
region_ind = "\n".join(("    " + ln) if ln.strip() else ln for ln in region.split("\n"))
guard = (
    "            _cache_hit = _expert_cache_get(args.get(\"task_config\"), task_name, now_seed)\n"
    "            if _cache_hit is not None:\n"
    "                _ci = _cache_hit.get(\"episode_info\") or {\"info\": _cache_hit.get(\"info\", {})}\n"
    "                episode_info = _ci\n"
    "                expert_plan_success = bool(_cache_hit.get(\"plan_success\"))\n"
    "                expert_task_success = bool(_cache_hit.get(\"task_success\"))\n"
    "                print(f\"[expert_cache] HIT seed={now_seed}\")\n"
    "            else:\n"
)
s = s[:i0] + guard + region_ind + s[i1_end:]

# 3) timing probes
rep("\n        if expert_check:\n            expert_plan_success = False",
    "\n        _exp_t0 = time.perf_counter()\n        if expert_check:\n            expert_plan_success = False")
rep("        args[\"render_freq\"] = render_freq\n        try:\n            task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n        except UnStableError:",
    "        _ACC[\"expert\"] += time.perf_counter() - _exp_t0\n        _pol_t0 = time.perf_counter()\n        args[\"render_freq\"] = render_freq\n        try:\n            task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)\n        except UnStableError:")
rep("        now_id += 1",
    "        _ACC[\"policy\"] += time.perf_counter() - _pol_t0\n        _et = _ACC.get(\"expert\")\n        _pt = _ACC.get(\"policy\")\n        print(f\"[phase_timing] expert_total={_et:.1f}s policy_total={_pt:.1f}s\")\n        now_id += 1")

ast.parse(s)
open(P, "w", encoding="utf-8").write(s)
print("patch applied OK")
