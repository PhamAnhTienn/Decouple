#!/usr/bin/env python
"""
Staged hyper-parameter tuning for DP on MOSEI (Screening -> Confirm -> Pick).

Workflow (chạy từng stage một):
    # Stage 1: Tune Loss Weights (CPA, Reconstruction, Preservation, Orthogonality)
    python tune_dp.py run     --stage 1            # screening at missing rate 0.5
    python tune_dp.py report  --stage 1 --marginal
    python tune_dp.py confirm --stage 1 --top 2    # re-run top-2 on all missing rates (0.0, 0.5, 0.7, 0.9)
    python tune_dp.py report  --stage 1
    python tune_dp.py pick    --stage 1 --tid <tid>   # winner becomes base for stage 2

    # Stage 2: Tune Uncertainty Dynamics & Prototypes (tau_c, tau_s, r_min, K_neighbors, nmb_prototypes)
    python tune_dp.py run     --stage 2
    python tune_dp.py report  --stage 2 --marginal
    python tune_dp.py confirm --stage 2 --top 2
    python tune_dp.py pick    --stage 2 --tid <tid>

    # Stage 3: Tune Optimization & Regularization (learning rates, weight decay, dropouts)
    python tune_dp.py run     --stage 3
    python tune_dp.py report  --stage 3
    python tune_dp.py confirm --stage 3 --top 2
    python tune_dp.py pick    --stage 3 --tid <tid>

    # Báo cáo kết quả cuối cùng cho cấu hình tốt nhất:
    python tune_dp.py final   --tid <tid>

All runs are resumable: finished runs are automatically skipped.
Checkpoints (.pth) are cleaned up after each trial to conserve disk space.
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Paths & Environment Configuration
# --------------------------------------------------------------------------------------
REPO = Path(os.environ.get("TUNE_REPO", Path(__file__).resolve().parent))
CONFIG_FILE = Path(os.environ.get("TUNE_CONFIG", REPO / "config" / "dp_mosei_config.json"))

# Automatically pick data dir (Kaggle or local)
default_kaggle_data = Path("/kaggle/input/datasets/phamanhtien/decouple")
if default_kaggle_data.exists():
    default_data_dir = str(default_kaggle_data)
else:
    default_data_dir = str(REPO / "data")

DATA_DIR = os.environ.get("TUNE_DATA", default_data_dir)
WORK = Path(os.environ.get("TUNE_WORK", REPO / "tune_work"))
RESULTS = WORK / "results.jsonl"
BEST_FILE = WORK / "best_config.json"
DATASET = "mosei"
ALL_MRS = [0.0, 0.5, 0.7, 0.9]

OBJ, OBJ_MODE = "Loss", "min"          # val L1 loss == criterion used for checkpoint selection
SHOW = ["Loss", "Has0_F1_score", "Non0_F1_score", "MAE", "Corr"]

FIXED = {}                              # e.g. {"bank_update_freq": 1}
CLI_FLAGS = {
    "K_neighbors": "--K_neighbors",
    "learning_rate": "--learning_rate",
    "batch_size": "--batch_size",
}

# Default values fallback when key is absent from json
DEFAULTS = {
    "tau_c": 1.0,
    "tau_s": 1.0,
    "r_min": 0.1,
    "K_neighbors": 10,
    "bank_update_freq": 1,
    "weight_decay": 0.001,
    "learning_rate": 0.00005,
    "bert_learning_rate": 1e-5,
    "lambda_recon_group": 0.1,
    "lambda_ort": 0.1,
    "lambda_cluster": 0.1,
    "lambda_preserve": 0.1,
    "nmb_prototypes": 7,
    "freeze_prototypes_niters": 600,
}
_M = object()


def _load_ref():
    ref = dict(DEFAULTS)
    try:
        ref.update(json.loads(CONFIG_FILE.read_text()))
    except Exception:
        pass
    return ref


REF = _load_ref()


def canon(ov):
    """Keep only keys that differ from reference config -> identical configs get identical ids."""
    return {k: ov[k] for k in sorted(ov) if REF.get(k, _M) != ov[k]}


def tid_of(ov):
    c = canon(ov)
    return hashlib.md5(json.dumps(c, sort_keys=True).encode()).hexdigest()[:8]


def F(key, vals):
    return [{key: v} for v in vals]


# --------------------------------------------------------------------------------------
# Stage Definitions (Lean & Focused, starting from Stage 1)
# --------------------------------------------------------------------------------------
STAGES = {
    # Stage 1: Core Loss Weights (CPA cluster, CRM recon, Specific preservation, Orthogonality)
    # OAT: One factor at a time against current baseline
    1: dict(desc="core loss weights (CPA / Recon / Preservation / Ortho)", kind="oat", screen=[0.5], factors=[
        F("lambda_cluster", [0.03, 0.3]),
        F("lambda_recon_group", [0.03, 0.3]),
        F("lambda_preserve", [0.03, 0.3, 1.0]),
        F("lambda_ort", [0.03, 0.3]),
    ]),

    # Stage 2: Uncertainty Dynamics & Prototypes
    # OAT on temperature, reliability floor, KNN neighbors, prototype counts
    2: dict(desc="uncertainty dynamics & prototypes", kind="oat", screen=[0.5], factors=[
        F("tau_c", [0.5, 2.0]),
        F("tau_s", [0.5, 2.0]),
        F("r_min", [0.0, 0.05, 0.2]),
        F("K_neighbors", [5, 20]),
        F("nmb_prototypes", [5, 10]),
    ]),

    # Stage 3: Optimization & Key Regularization
    # Random sampling across high-impact training params
    3: dict(desc="optimisation & main regularisation", kind="random", n=8, screen=[0.5], factors=[
        F("learning_rate", [3e-5, 5e-5, 1e-4]),
        F("bert_learning_rate", [5e-6, 1e-5, 2e-5]),
        F("weight_decay", [1e-4, 1e-3, 1e-2]),
        F("output_dropout", [0.3, 0.5]),
        F("attn_dropout", [0.2, 0.4]),
    ]),
}


# --------------------------------------------------------------------------------------
# Bookkeeping & Candidate Generation
# --------------------------------------------------------------------------------------
def load_results():
    out = {}
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    out[(r["tid"], r["mr"], r["seed"])] = r
                except Exception:
                    pass
    return out


def load_best():
    return json.loads(BEST_FILE.read_text()) if BEST_FILE.exists() else {}


def meta_path(stage):
    return WORK / f"stage{stage}.json"


def load_meta(stage):
    return json.loads(meta_path(stage).read_text())


def make_candidates(spec, base, rng):
    b = canon(base)
    base_tid = tid_of(b)
    out = {base_tid: b}
    kind = spec["kind"]
    if kind == "fixed":
        return out
    if kind == "oat":
        for fac in spec["factors"]:
            for frag in fac:
                c = canon({**base, **frag})
                out.setdefault(tid_of(c), c)
    elif kind == "grid":
        for combo in itertools.product(*spec["factors"]):
            frag = {k: v for d in combo for k, v in d.items()}
            c = canon({**base, **frag})
            out.setdefault(tid_of(c), c)
    else:  # random
        for _ in range(5000):
            if len(out) >= spec["n"] + 1:
                break
            frag = {}
            for fac in spec["factors"]:
                frag.update(rng.choice(fac))
            c = canon({**base, **frag})
            out.setdefault(tid_of(c), c)
    return out


# --------------------------------------------------------------------------------------
# Execution: Subprocess main.py per trial run
# --------------------------------------------------------------------------------------
def run_one(stage, tid, ov, mr, seed, results, dry=False):
    key = (tid, mr, seed)
    if results.get(key, {}).get("status") == "ok":
        return
    run_dir = WORK / f"stage{stage}" / tid / f"mr{mr}_s{seed}"
    cmd = [
        sys.executable, "main.py",
        "--data_missing",
        "--model", "dp",
        "--config_file", str(CONFIG_FILE),
        "--data_dir", str(DATA_DIR),
        "--dataset", DATASET,
        "--mode", "train",
        "--seeds", str(seed),
        "--model_save_dir", str(run_dir / "pt"),
        "--res_save_dir", str(run_dir / "res"),
        "--log_dir", str(run_dir / "log"),
        "--missing_rate_eval_test", str(mr),
    ]

    merged_ov = {**FIXED, **ov}
    for k, v in merged_ov.items():
        if k in CLI_FLAGS:
            cmd += [CLI_FLAGS[k], str(v)]
        else:
            cmd += ["--config_override", f"{k}={v}"]

    print(f"\n[stage{stage}] tid={tid} mr={mr} seed={seed} | changes={ov}")
    if dry:
        print("   cmd:", " ".join(cmd))
        return

    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(run_dir / "stdout.txt", "w") as f:
        rc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode

    rec = dict(
        stage=stage,
        tid=tid,
        ov=ov,
        mr=mr,
        seed=seed,
        status="ok" if rc == 0 else "fail",
        minutes=round((time.time() - t0) / 60, 1),
        val={},
        test={},
    )

    vf = run_dir / "res" / f"best_val_seed{seed}.json"
    if vf.exists():
        try:
            rec["val"] = json.loads(vf.read_text())
        except Exception:
            rec["status"] = "fail"
    else:
        rec["status"] = "fail"

    csv = run_dir / "res" / "normal" / f"{DATASET}.csv"
    if csv.exists():
        try:
            df = pd.read_csv(csv)
            row = df.iloc[-1]
            rec["test"] = {c[:-5]: float(row[c]) for c in df.columns if c.endswith(" Mean")}
        except Exception:
            pass

    # Free up disk: remove model checkpoint weights (~0.5GB each)
    shutil.rmtree(run_dir / "pt", ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    results[key] = rec
    val = rec["val"].get(OBJ, "N/A")
    print(f"    -> Status: {rec['status']} | {OBJ}={val} | Time: {rec['minutes']} min")


# --------------------------------------------------------------------------------------
# Scoring & Reporting
# --------------------------------------------------------------------------------------
def per_mr(results, tid, key=OBJ):
    d = {}
    for (t, mr, _), r in results.items():
        if t == tid and r.get("status") == "ok" and key in r.get("val", {}):
            d.setdefault(mr, []).append(r["val"][key])
    return {mr: float(np.mean(v)) for mr, v in d.items()}


def score(results, tid, mrs=None):
    d = per_mr(results, tid)
    if mrs is not None:
        d = {m: d[m] for m in mrs if m in d}
    return float(np.mean(list(d.values()))) if d else float("nan")


def ranked(results, meta, mrs=None):
    sc = {t: score(results, t, mrs) for t in meta["cands"]}
    sc = {t: s for t, s in sc.items() if not np.isnan(s)}
    return sorted(sc, key=sc.get, reverse=(OBJ_MODE == "max"))


def fmt_ov(ov, base_ov):
    d = {k: v for k, v in ov.items() if base_ov.get(k, _M) != v}
    for k in base_ov:
        if k not in ov:
            d[k] = REF.get(k, "default")
    return ", ".join(f"{k}={v}" for k, v in d.items()) or "(base)"


def cmd_report(a):
    meta, res = load_meta(a.stage), load_results()
    cands, base_tid = meta["cands"], meta["base_tid"]
    base_ov = cands.get(base_tid, {})
    rows = []
    for tid, ov in cands.items():
        pm = per_mr(res, tid)
        if not pm:
            continue
        s = float(np.mean(list(pm.values())))
        bs = score(res, base_tid, list(pm))
        row = {
            "tid": tid + ("*" if tid == base_tid else ""),
            "mrs": len(pm),
            "score": s,
            "d_vs_base": s - bs if not np.isnan(bs) else 0.0,
        }
        row.update({f"mr{m}": pm[m] for m in sorted(pm)})
        for k in SHOW:
            if k != OBJ:
                v = per_mr(res, tid, k)
                if v:
                    row[k] = float(np.mean(list(v.values())))
        row["changes"] = fmt_ov(ov, base_ov)
        rows.append(row)
    if not rows:
        print("No completed runs yet for stage", a.stage)
        return
    df = pd.DataFrame(rows).sort_values("score", ascending=(OBJ_MODE == "min"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(f"\n==========================================================================================")
    print(f"STAGE {a.stage}: {STAGES[a.stage]['desc']} | Objective: val {OBJ} ({OBJ_MODE}) | * = base")
    print(f"==========================================================================================")
    full = df[df["mrs"] == len(ALL_MRS)]
    part = df[df["mrs"] < len(ALL_MRS)]
    if len(full):
        print(f"\n[Confirmed across all {len(ALL_MRS)} missing rates (Score = Mean across MRs)]:")
        print(full.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if len(part):
        print(f"\n[Screening runs (Score on screened MRs; d_vs_base computed on matched MRs)]:")
        print(part.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Seed noise check (if >= 2 seeds available)
    noise = []
    for tid in cands:
        by = {}
        for (t, mr, sd), r in res.items():
            if t == tid and r.get("status") == "ok" and OBJ in r.get("val", {}):
                by.setdefault(mr, []).append(r["val"][OBJ])
        for mr, v in by.items():
            if len(v) >= 2:
                noise.append((tid, mr, len(v), np.mean(v), np.std(v)))
    if noise:
        print(f"\nSeed variance (val {OBJ}):")
        for tid, mr, n, m, s in noise:
            print(f"  tid={tid} mr={mr} (n={n}): {m:.4f} +- {s:.4f}")

    if a.marginal:
        groups = {}
        for tid, ov in cands.items():
            s = score(res, tid)
            if np.isnan(s):
                continue
            keys = set(ov) | set(base_ov)
            for k in keys:
                groups.setdefault(k, {}).setdefault(str(ov.get(k, REF.get(k, "default"))), []).append(s)
        print("\nMarginal mean score per hyperparameter value (lower is better):")
        for k, g in sorted(groups.items()):
            if len(g) > 1:
                print(f"  {k}: " + " | ".join(f"{v}: {np.mean(x):.4f} (n={len(x)})" for v, x in sorted(g.items())))


# --------------------------------------------------------------------------------------
# CLI Subcommands
# --------------------------------------------------------------------------------------
def cmd_run(a):
    WORK.mkdir(parents=True, exist_ok=True)
    if a.stage not in STAGES:
        print(f"Error: Stage {a.stage} not defined. Available stages: {list(STAGES.keys())}")
        return
    spec = STAGES[a.stage]
    if meta_path(a.stage).exists() and not a.regen:
        meta = load_meta(a.stage)
    else:
        base = load_best()
        base_c = canon(base)
        cands = make_candidates(spec, base, random.Random(1000 + a.stage))
        meta = dict(base_tid=tid_of(base_c), cands=cands)
        meta_path(a.stage).write_text(json.dumps(meta, indent=2))

    runs = spec.get("runs") or [(mr, a.seed) for mr in spec["screen"]]
    res = load_results()
    total = len(meta["cands"]) * len(runs)
    print(f"\n>>> Running Stage {a.stage} ({spec['desc']}):")
    print(f"    {len(meta['cands'])} trials x {len(runs)} runs = {total} total runs (skipping completed runs)")

    for tid, ov in meta["cands"].items():
        for mr, seed in runs:
            run_one(a.stage, tid, ov, mr, seed, res, a.dry)

    if not a.dry:
        cmd_report(argparse.Namespace(stage=a.stage, marginal=spec["kind"] != "fixed"))


def cmd_confirm(a):
    if a.stage not in STAGES:
        print(f"Error: Stage {a.stage} not defined.")
        return
    meta, res = load_meta(a.stage), load_results()
    spec = STAGES[a.stage]
    screen = spec.get("screen") or [0.5]
    top = ranked(res, meta, screen)[: a.top]
    mrs = a.mrs or ALL_MRS
    print(f"\n>>> Confirming top-{a.top} candidates: {top} on MRs={mrs} seeds={a.seeds}")
    for tid in top:
        for seed in a.seeds:
            for mr in mrs:
                run_one(a.stage, tid, meta["cands"][tid], mr, seed, res, a.dry)
    if not a.dry:
        cmd_report(argparse.Namespace(stage=a.stage, marginal=False))


def cmd_pick(a):
    if a.stage not in STAGES:
        print(f"Error: Stage {a.stage} not defined.")
        return
    meta = load_meta(a.stage)
    if a.tid not in meta["cands"]:
        print(f"Error: tid '{a.tid}' not found in stage {a.stage} candidates.")
        return
    ov = meta["cands"][a.tid]
    WORK.mkdir(parents=True, exist_ok=True)
    BEST_FILE.write_text(json.dumps(ov, indent=2))
    try:
        full = {**json.loads(CONFIG_FILE.read_text()), **ov}
    except Exception:
        full = ov
    (WORK / "best_config_full.json").write_text(json.dumps(full, indent=2))
    print(f"\n[Picked {a.tid} as new base config for subsequent stages]:")
    print("Diff vs reference:", json.dumps(ov, indent=2))
    print("Saved full config ->", WORK / "best_config_full.json")


def cmd_final(a):
    res = load_results()
    runs = [r for (t, _, _), r in res.items() if t == a.tid and r.get("status") == "ok"]
    if not runs:
        print(f"No successful runs found for tid {a.tid}")
        return
    for split in ("val", "test"):
        print(f"\n== {a.tid} Final {split.upper()} Metrics (mean +- std over seeds) ==")
        for mr in sorted({r["mr"] for r in runs}):
            rs = [r[split] for r in runs if r["mr"] == mr]
            parts = []
            for k in SHOW:
                v = [x[k] for x in rs if k in x]
                if v:
                    parts.append(f"{k}={np.mean(v):.4f}+-{np.std(v):.4f}")
            print(f"  mr={mr} (n={len(rs)}): " + "  ".join(parts))


def main():
    p = argparse.ArgumentParser(description="Staged Tuning for DP on MOSEI")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run screening for a stage")
    r.add_argument("--stage", type=int, required=True, help="Stage index (1, 2, or 3)")
    r.add_argument("--seed", type=int, default=1111)
    r.add_argument("--dry", action="store_true", help="Print commands without executing")
    r.add_argument("--regen", action="store_true", help="Regenerate candidates list")

    c = sub.add_parser("confirm", help="Confirm top candidates across all missing rates")
    c.add_argument("--stage", type=int, required=True)
    c.add_argument("--top", type=int, default=2)
    c.add_argument("--mrs", type=float, nargs="+")
    c.add_argument("--seeds", type=int, nargs="+", default=[1111])
    c.add_argument("--dry", action="store_true")

    rp = sub.add_parser("report", help="Display score table and marginal analysis")
    rp.add_argument("--stage", type=int, required=True)
    rp.add_argument("--marginal", action="store_true", help="Show marginal impact per hyperparameter")

    pk = sub.add_parser("pick", help="Pick winner tid to be base for subsequent stages")
    pk.add_argument("--stage", type=int, required=True)
    pk.add_argument("--tid", required=True)

    fn = sub.add_parser("final", help="Print final summary for a trial id")
    fn.add_argument("--tid", required=True)

    a = p.parse_args()
    {"run": cmd_run, "confirm": cmd_confirm, "report": cmd_report, "pick": cmd_pick, "final": cmd_final}[a.cmd](a)


if __name__ == "__main__":
    main()
