"""Load a run_study.py output directory: <study>/<stage>/<timestamp>/*.csv."""
import json
import sys
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

KEY = ["trial", "model", "backend"]   # "backend" = config label


class Stage:
    def __init__(self, path):
        self.path = Path(path)

    def csv(self, name):
        f = self.path / name
        return pd.read_csv(f, low_memory=False) if f.exists() else pd.DataFrame()

    @cached_property
    def sessions(self):
        return self.csv("sessions.csv")

    @cached_property
    def runs(self):
        return self.csv("runs.csv")

    @cached_property
    def experiments(self):
        f = self.path / "experiments.jsonl"
        return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] if f.exists() else []

    @cached_property
    def env(self):
        f = self.path / "env.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def ok_keys(self):
        s = self.sessions
        return s[s["status"] == "ok"][KEY] if len(s) else pd.DataFrame(columns=KEY)

    def ok_runs(self, phase=None):
        r = self.runs.merge(self.ok_keys(), on=KEY) if len(self.runs) else self.runs
        return r[r["phase"] == phase] if phase and len(r) else r


class Study:
    def __init__(self, root):
        self.root = Path(root)
        self.stages = {}
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            # readback_micro has no sessions (no model); its data is readback_micro.csv.
            subs = [s for s in d.iterdir() if s.is_dir() and
                    ((s / "sessions.csv").exists() or (s / "readback_micro.csv").exists())]
            if subs:
                self.stages[d.name] = Stage(max(subs, key=lambda s: s.stat().st_mtime))

    def __getitem__(self, name):
        return self.stages.get(name)

    def has(self, name):
        return name in self.stages

    @cached_property
    def registry(self):
        st = next(iter(self.stages.values()))
        reg = Path(st.env["args"]["models"])
        if not reg.is_absolute():
            reg = ROOT / reg
        return {m["name"]: m for m in json.loads(reg.read_text())["models"]}

    @cached_property
    def features(self):
        """Static per-model features (MACs, weight bytes, nodes, WebGL rule), cached on disk."""
        cache = self.root / "figures" / "model_features.csv"
        if cache.exists():
            return pd.read_csv(cache)
        from predict import static_features, webgl_op_table
        rules = webgl_op_table()
        reg = Path(next(iter(self.stages.values())).env["args"]["models"])
        public = (reg if reg.is_absolute() else ROOT / reg).resolve().parents[1]
        f = pd.DataFrame([static_features(m, public, rules) for m in self.registry.values()])
        cache.parent.mkdir(parents=True, exist_ok=True)
        f.to_csv(cache, index=False)
        return f

    def steady(self, stage="latency"):
        """Per (model, config): measured latencies pooled over trials, plus the median."""
        st = self[stage]
        r = st.ok_runs("measure")
        return r.groupby(["model", "backend"])["dur_ms"].apply(np.asarray)
