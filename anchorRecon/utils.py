#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import time
import json
import csv
import statistics
import contextlib
import os
from collections import defaultdict, OrderedDict



def get_image_names(in_folder, image_extensions=[".jpg", ".png", ".jpeg"]):
    return [
        f
        for f in os.listdir(in_folder)
        if os.path.splitext(f)[-1].lower() in image_extensions
    ]
















## Camera/triangulation/projection functions
def principal_point_xy(width, height, cx=None, cy=None):
    if cx is None:
        cx = (width - 1) / 2
    if cy is None:
        cy = (height - 1) / 2
    return float(cx), float(cy)


def principal_point_tensor(width, height, cx=None, cy=None, device="cuda"):
    cx, cy = principal_point_xy(width, height, cx, cy)
    return torch.tensor([cx, cy], device=device, dtype=torch.float32)




def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))




def depth2points(uv, depth, f, centre):
    xyz = torch.cat([(uv[..., :2] - centre) / f, torch.ones_like(uv[..., 0:1])], dim=-1)
    return depth * xyz




def make_torch_sampler(uv, width, height):
    """
    Converts OpenCV UV coordinates to a sampler for torch's grid_sample.
    To be used with align_corners=True
    """
    sampler = uv.clone()  # + 0.5
    sampler[..., 0] = sampler[..., 0] * (2.0 / (width - 1)) - 1.0
    sampler[..., 1] = sampler[..., 1] * (2.0 / (height - 1)) - 1.0
    return sampler




def pts2px(xyz, f, centre):
    return f * xyz[..., :2] / xyz[..., 2:3] + centre


def sixD2mtx(r):
    b1 = r[..., 0]
    b1 = b1 / torch.norm(b1, dim=-1, keepdim=True)
    b2 = r[..., 1] - torch.sum(b1 * r[..., 1], dim=-1, keepdim=True) * b1
    b2 = b2 / torch.norm(b2, dim=-1, keepdim=True)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def mtx2sixD(R):
    return R[..., :2].clone()


## Visualization functions









## Pose alignment and evaluation functions


## From https://github.com/chenhsuanlin/bundle-adjusting-NeRF
# BARF: Bundle-Adjusting Neural Radiance Fields
# Copyright (c) 2021 Chen-Hsuan Lin
# Under the MIT License.
# Modified to interface with our pose format 




# ---------------------------------------------------------------------------
# Timing profiler
# ---------------------------------------------------------------------------
#
# Lightweight CUDA-aware profiler used to break down per-frame and per-submap
# runtime in the anchored pose pass. Designed to be:
#
#   * zero-overhead when disabled (``with prof.stage("x"): ...`` becomes a
#     no-op context manager that just yields),
#   * nestable (stages use a stack, sub-stage time is *also* counted in the
#     parent's running time, exactly like the standard cProfile cumulative
#     time),
#   * useful for paper reporting (per-stage mean / median / std, plus
#     per-frame CSV and global-summary JSON).
#
# Typical use::
#
#     prof = TimingProfiler(enabled=True, sync_cuda=True)
#     set_global_profiler(prof)
#     ...
#     with prof.frame(frame_id=k, name=fname, submap=s, is_anchor=a):
#         with prof.stage("io"): image, info = dataset.getnext()
#         with prof.stage("detector"): kpts = detector(image)
#         ...
#     prof.report(out_dir="results/profiling/")
#
# Modules that need to add sub-stage probes can do::
#
#     from utils import get_global_profiler
#     with get_global_profiler().stage("match.loma"): ...
#
# When ``enabled=False`` (or no profiler has been registered), all the
# context managers fall through to a no-op singleton.


class _NoopStage(contextlib.AbstractContextManager):
    """Cheap no-op context manager used when timing is disabled."""

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


_NOOP_STAGE = _NoopStage()


class _NoopProfiler:
    """Profiler stub returned when no profiler has been installed.

    Exposes the same surface as :class:`TimingProfiler` but skips all work.
    """

    enabled = False

    def stage(self, name):  # noqa: D401 - context manager
        return _NOOP_STAGE

    def frame(self, **meta):  # noqa: D401 - context manager
        return _NOOP_STAGE

    def begin_frame(self, **meta):
        return None

    def end_frame(self, **extra_meta):
        return None

    def begin_submap(self, submap_id, **meta):
        return None

    def end_submap(self):
        return None

    def add(self, name, dt, count=1):
        return None

    def incr_counter(self, name, n=1):
        return None

    def record_event(self, name, value):
        return None

    def report(self, out_dir=None, label=None):
        return None

    def flush_pending(self):
        return None


_NOOP_PROFILER = _NoopProfiler()


_GLOBAL_PROFILER = _NOOP_PROFILER


def set_global_profiler(profiler):
    """Install the profiler that ``get_global_profiler`` will return.

    Pass ``None`` to disable cross-module probing again.
    """
    global _GLOBAL_PROFILER
    _GLOBAL_PROFILER = profiler if profiler is not None else _NOOP_PROFILER


def get_global_profiler():
    """Return the active profiler (no-op singleton if none installed)."""
    return _GLOBAL_PROFILER


class _StageContext(contextlib.AbstractContextManager):
    """Active-profiler context manager that times one stage."""

    __slots__ = ("_profiler", "_name", "_t0")

    def __init__(self, profiler, name):
        self._profiler = profiler
        self._name = name
        self._t0 = 0.0

    def __enter__(self):
        prof = self._profiler
        if prof.sync_cuda:
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        prof._stage_stack.append(self._name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        prof = self._profiler
        if prof.sync_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._t0
        if prof._stage_stack and prof._stage_stack[-1] == self._name:
            prof._stage_stack.pop()
        prof.add(self._name, dt, count=1)
        return False


class _FrameContext(contextlib.AbstractContextManager):
    """Active-profiler context manager that wraps one frame."""

    __slots__ = ("_profiler", "_meta")

    def __init__(self, profiler, meta):
        self._profiler = profiler
        self._meta = meta

    def __enter__(self):
        self._profiler.begin_frame(**self._meta)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._profiler.end_frame()
        return False


class TimingProfiler:
    """Hierarchical wall-clock + CUDA-aware profiler.

    Parameters
    ----------
    enabled : bool
        Master switch; when False every ``stage()`` is essentially a no-op
        (the context manager still runs but no timing is recorded).
    sync_cuda : bool
        Insert ``torch.cuda.synchronize()`` around every ``stage()`` enter and
        exit. Required for accurate GPU timings; adds a few µs per call.
    track_per_frame : bool
        Keep a per-frame breakdown in memory (used by ``save_csv``).
    """

    def __init__(self, enabled=True, sync_cuda=True, track_per_frame=True):
        self.enabled = bool(enabled)
        self.sync_cuda = bool(sync_cuda) and torch.cuda.is_available()
        self.track_per_frame = bool(track_per_frame)

        # Aggregated stats across the whole run.
        self.stage_total = defaultdict(float)   # cumulative seconds per stage
        self.stage_count = defaultdict(int)     # number of times entered
        self.counters = defaultdict(int)        # arbitrary integer counters

        # Per-frame and per-submap breakdowns (only if track_per_frame).
        self.frame_records = []                 # list[dict] - one per frame
        self.submap_records = []                # list[dict] - one per submap

        # Per-frame state - filled while a frame is "open".
        self._frame_open = False
        self._frame_meta = {}
        self._frame_stages = defaultdict(float)
        self._frame_counts = defaultdict(int)
        self._frame_t0 = 0.0
        self._frame_index = 0

        # Per-submap state.
        self._submap_open = False
        self._submap_id = None
        self._submap_meta = {}
        self._submap_t0 = 0.0
        self._submap_stages = defaultdict(float)
        self._submap_frame_count = 0
        self._submap_keyframe_count = 0

        # Used by stage context managers.
        self._stage_stack = []

    # ----- low-level recorders -------------------------------------------------
    def add(self, name, dt, count=1):
        if not self.enabled:
            return
        self.stage_total[name] += dt
        self.stage_count[name] += count
        if self._frame_open:
            self._frame_stages[name] += dt
            self._frame_counts[name] += count
        if self._submap_open:
            self._submap_stages[name] += dt

    def incr_counter(self, name, n=1):
        if not self.enabled:
            return
        self.counters[name] += n
        if self._frame_open:
            self._frame_counts[f"counter::{name}"] += n

    # ----- context managers ----------------------------------------------------
    def stage(self, name):
        if not self.enabled:
            return _NOOP_STAGE
        return _StageContext(self, name)

    def frame(self, **meta):
        if not self.enabled or not self.track_per_frame:
            return _NOOP_STAGE
        return _FrameContext(self, meta)

    # ----- explicit frame boundaries -------------------------------------------
    def begin_frame(self, **meta):
        if not self.enabled or not self.track_per_frame:
            return
        if self._frame_open:
            # Auto-close stale frame; should not happen in normal flow.
            self.end_frame()
        if self.sync_cuda:
            torch.cuda.synchronize()
        self._frame_open = True
        self._frame_meta = dict(meta)
        self._frame_stages = defaultdict(float)
        self._frame_counts = defaultdict(int)
        self._frame_t0 = time.perf_counter()
        self._frame_index += 1

    def end_frame(self, **extra_meta):
        if not self.enabled or not self.track_per_frame or not self._frame_open:
            return None
        if self.sync_cuda:
            torch.cuda.synchronize()
        wall = time.perf_counter() - self._frame_t0
        record = {
            "idx": self._frame_index,
            "wall": wall,
            "meta": {**self._frame_meta, **extra_meta},
            "stages": dict(self._frame_stages),
            "counts": dict(self._frame_counts),
        }
        self.frame_records.append(record)
        self.stage_total["frame"] += wall
        self.stage_count["frame"] += 1
        if self._submap_open:
            self._submap_frame_count += 1
            if record["meta"].get("is_keyframe", False):
                self._submap_keyframe_count += 1
        self._frame_open = False
        self._frame_meta = {}
        self._frame_stages = defaultdict(float)
        self._frame_counts = defaultdict(int)
        return record

    # ----- per-submap boundaries -----------------------------------------------
    def begin_submap(self, submap_id, **meta):
        if not self.enabled:
            return
        if self._submap_open:
            self.end_submap()
        if self.sync_cuda:
            torch.cuda.synchronize()
        self._submap_open = True
        self._submap_id = submap_id
        self._submap_meta = dict(meta)
        self._submap_t0 = time.perf_counter()
        self._submap_stages = defaultdict(float)
        self._submap_frame_count = 0
        self._submap_keyframe_count = 0

    def end_submap(self):
        if not self.enabled or not self._submap_open:
            return None
        if self.sync_cuda:
            torch.cuda.synchronize()
        wall = time.perf_counter() - self._submap_t0
        record = {
            "submap": self._submap_id,
            "wall": wall,
            "meta": dict(self._submap_meta),
            "stages": dict(self._submap_stages),
            "n_frames": self._submap_frame_count,
            "n_keyframes": self._submap_keyframe_count,
        }
        self.submap_records.append(record)
        self._submap_open = False
        self._submap_id = None
        self._submap_meta = {}
        self._submap_stages = defaultdict(float)
        return record

    # ----- summarization & export ---------------------------------------------
    def _frame_stage_series(self, stage_name, only_keyframes=None, only_present=True):
        """Return a list of per-frame durations for ``stage_name``.

        ``only_keyframes``: True -> keyframe rows only, False -> non-keyframe
        rows only, None -> all rows.
        ``only_present``: when True (default), skip frames where the stage was
        never entered (avoids polluting mean/median with zeros when a stage
        only fires on a subset of frames, e.g. ``incremental.*`` only on
        keyframes).
        """
        out = []
        for rec in self.frame_records:
            if only_keyframes is True and not rec["meta"].get("is_keyframe", False):
                continue
            if only_keyframes is False and rec["meta"].get("is_keyframe", False):
                continue
            if stage_name in rec["stages"]:
                out.append(rec["stages"][stage_name])
            elif not only_present:
                out.append(0.0)
        return out

    @staticmethod
    def _stat(values):
        if not values:
            return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        if len(values) == 1:
            v = values[0]
            return {"n": 1, "mean": v, "median": v, "std": 0.0, "min": v, "max": v}
        return {
            "n": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }

    def summary(self):
        """Return a structured summary dict suitable for paper tables."""
        if not self.enabled:
            return {"enabled": False}

        all_stages = sorted(self.stage_total.keys())
        per_stage = OrderedDict()
        for stage in all_stages:
            per_stage[stage] = {
                "calls": self.stage_count[stage],
                "total": self.stage_total[stage],
                "avg_ms": (self.stage_total[stage] / self.stage_count[stage] * 1000.0)
                if self.stage_count[stage] > 0
                else 0.0,
            }

        wall_per_frame_kf = [r["wall"] for r in self.frame_records if r["meta"].get("is_keyframe", False)]
        wall_per_frame_nokf = [r["wall"] for r in self.frame_records if not r["meta"].get("is_keyframe", False)]
        wall_all = [r["wall"] for r in self.frame_records]

        per_frame_stage_stats = OrderedDict()
        if self.frame_records:
            tracked = sorted({s for r in self.frame_records for s in r["stages"].keys()})
            for stage in tracked:
                per_frame_stage_stats[stage] = {
                    "all": self._stat(self._frame_stage_series(stage)),
                    "keyframes": self._stat(self._frame_stage_series(stage, only_keyframes=True)),
                    "non_keyframes": self._stat(self._frame_stage_series(stage, only_keyframes=False)),
                }

        return {
            "enabled": True,
            "n_frames_total": len(self.frame_records),
            "n_keyframes_total": sum(
                1 for r in self.frame_records if r["meta"].get("is_keyframe", False)
            ),
            "n_submaps": len(self.submap_records),
            "wall_per_frame": {
                "all": self._stat(wall_all),
                "keyframes": self._stat(wall_per_frame_kf),
                "non_keyframes": self._stat(wall_per_frame_nokf),
            },
            "per_stage": per_stage,
            "per_frame_stage": per_frame_stage_stats,
            "counters": dict(self.counters),
            "submaps": [
                {
                    "submap": r["submap"],
                    "wall": r["wall"],
                    "n_frames": r["n_frames"],
                    "n_keyframes": r["n_keyframes"],
                    "stages": r["stages"],
                }
                for r in self.submap_records
            ],
        }

    def format_brief(self, max_stages=8):
        """Compact one-paragraph summary; useful as a per-submap progress print."""
        if not self.enabled or not self.frame_records:
            return "[profiler] no frames recorded yet"
        s = self.summary()
        wpf = s["wall_per_frame"]
        stages_sorted = sorted(
            s["per_stage"].items(),
            key=lambda kv: kv[1]["total"],
            reverse=True,
        )
        top_lines = []
        for name, info in stages_sorted:
            if name == "frame":
                continue
            if len(top_lines) >= max_stages:
                break
            top_lines.append(f"{name}={info['avg_ms']:.0f}ms")
        line1 = (
            f"[profiler/brief] frames={s['n_frames_total']} "
            f"(kf={s['n_keyframes_total']}) "
            f"wall/frame all={wpf['all']['mean']*1e3:.0f}ms "
            f"kf={wpf['keyframes']['mean']*1e3:.0f}ms "
            f"!kf={wpf['non_keyframes']['mean']*1e3:.0f}ms"
        )
        line2 = "[profiler/brief] top: " + " | ".join(top_lines)
        return line1 + "\n" + line2

    def format_table(self):
        """Pretty ASCII table for stdout printing."""
        if not self.enabled or not self.frame_records:
            return "[profiler] no frames recorded"

        s = self.summary()
        lines = []
        lines.append(
            "\n================ Timing profiler summary ================"
        )
        lines.append(
            f"frames={s['n_frames_total']} (keyframes={s['n_keyframes_total']}), "
            f"submaps={s['n_submaps']}"
        )

        wpf = s["wall_per_frame"]
        lines.append(
            "Wall per frame (ms): "
            f"all  mean={wpf['all']['mean']*1e3:7.1f}  median={wpf['all']['median']*1e3:7.1f}  "
            f"std={wpf['all']['std']*1e3:7.1f}  max={wpf['all']['max']*1e3:7.1f}"
        )
        if wpf["keyframes"]["n"]:
            lines.append(
                "                     "
                f"kf   mean={wpf['keyframes']['mean']*1e3:7.1f}  median={wpf['keyframes']['median']*1e3:7.1f}  "
                f"std={wpf['keyframes']['std']*1e3:7.1f}  max={wpf['keyframes']['max']*1e3:7.1f}  "
                f"n={wpf['keyframes']['n']}"
            )
        if wpf["non_keyframes"]["n"]:
            lines.append(
                "                     "
                f"!kf  mean={wpf['non_keyframes']['mean']*1e3:7.1f}  median={wpf['non_keyframes']['median']*1e3:7.1f}  "
                f"std={wpf['non_keyframes']['std']*1e3:7.1f}  max={wpf['non_keyframes']['max']*1e3:7.1f}  "
                f"n={wpf['non_keyframes']['n']}"
            )

        lines.append("")
        lines.append(
            "{:<28s} {:>7s} {:>10s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
                "stage", "calls", "total(s)", "avg(ms)", "kf(ms)", "!kf(ms)", "share"
            )
        )
        lines.append("-" * 86)
        total_wall = sum(r["wall"] for r in self.frame_records)
        for stage, info in s["per_stage"].items():
            # 'frame' is the synthetic per-frame wall already shown in
            # wall_per_frame above; skip it in the per-stage breakdown to
            # avoid duplicate / confusing rows.
            if stage == "frame":
                continue
            has_per_frame = stage in s["per_frame_stage"]
            kf_avg = (
                s["per_frame_stage"][stage]["keyframes"]["mean"] * 1e3
                if has_per_frame
                else 0.0
            )
            nokf_avg = (
                s["per_frame_stage"][stage]["non_keyframes"]["mean"] * 1e3
                if has_per_frame
                else 0.0
            )
            kf_n = (
                s["per_frame_stage"][stage]["keyframes"]["n"]
                if has_per_frame
                else 0
            )
            nokf_n = (
                s["per_frame_stage"][stage]["non_keyframes"]["n"]
                if has_per_frame
                else 0
            )
            kf_str = f"{kf_avg:.2f}" if kf_n > 0 else "  -  "
            nokf_str = f"{nokf_avg:.2f}" if nokf_n > 0 else "  -  "
            share = info["total"] / total_wall * 100.0 if total_wall > 0 else 0.0
            lines.append(
                "{:<28s} {:>7d} {:>10.2f} {:>9.2f} {:>9s} {:>9s} {:>8.1f}%".format(
                    stage,
                    info["calls"],
                    info["total"],
                    info["avg_ms"],
                    kf_str,
                    nokf_str,
                    share,
                )
            )

        if self.submap_records:
            lines.append("")
            lines.append(
                "{:<8s} {:>9s} {:>9s} {:>9s} {:>10s} {:>10s}".format(
                    "submap", "frames", "kf", "wall(s)", "boot(s)", "finalBA(s)"
                )
            )
            lines.append("-" * 60)
            for r in self.submap_records:
                lines.append(
                    "{:<8s} {:>9d} {:>9d} {:>9.2f} {:>10.2f} {:>10.2f}".format(
                        str(r["submap"]),
                        r["n_frames"],
                        r["n_keyframes"],
                        r["wall"],
                        r["stages"].get("bootstrap.total", 0.0),
                        r["stages"].get("final_ba.total", 0.0),
                    )
                )

        if self.counters:
            lines.append("")
            lines.append("counters:")
            for k, v in sorted(self.counters.items()):
                lines.append(f"  {k:<32s} {v}")

        lines.append(
            "=========================================================\n"
        )
        return "\n".join(lines)

    def save_csv(self, path):
        """Dump per-frame breakdown as CSV (one row per frame)."""
        if not self.enabled or not self.frame_records:
            return None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        all_stages = sorted({k for r in self.frame_records for k in r["stages"].keys()})
        meta_keys = sorted({k for r in self.frame_records for k in r["meta"].keys()})
        fieldnames = ["idx", "wall_ms"] + [f"meta.{k}" for k in meta_keys] + [
            f"stage.{s}_ms" for s in all_stages
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.frame_records:
                row = {"idx": r["idx"], "wall_ms": r["wall"] * 1000.0}
                for k in meta_keys:
                    row[f"meta.{k}"] = r["meta"].get(k, "")
                for s in all_stages:
                    row[f"stage.{s}_ms"] = r["stages"].get(s, 0.0) * 1000.0
                writer.writerow(row)
        return path

    def save_json(self, path):
        """Dump global summary as JSON for paper tables / plots."""
        if not self.enabled:
            return None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2, default=float)
        return path

    def report(self, out_dir=None, label=None):
        """Print summary table; optionally write CSV+JSON under ``out_dir``."""
        if not self.enabled:
            return None
        print(self.format_table())
        out = {"label": label}
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, "per_frame_timings.csv")
            json_path = os.path.join(out_dir, "summary.json")
            self.save_csv(csv_path)
            self.save_json(json_path)
            out["csv"] = csv_path
            out["json"] = json_path
            print(f"[profiler] wrote {csv_path}")
            print(f"[profiler] wrote {json_path}")
        return out

    def flush_pending(self):
        """Close any frame/submap left open by an early exit / exception."""
        if not self.enabled:
            return
        if self._frame_open:
            self.end_frame()
        if self._submap_open:
            self.end_submap()

