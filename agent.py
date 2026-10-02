"""Agenthon 2026 Track 2 — the submission agent, in one file.

Single file on purpose. The image needs no package layout and imports nothing but numpy, pandas
and pyarrow, so there is no import path to get wrong inside a scored run — and a unit that fails
scores 4.0 and stays in the leaderboard denominator, which costs more than any modelling gain
measured across four pre-registered rounds (see STATUS.md).

What it computes is M0's own construction — the official text-blind baseline specified in the
kit's docs/M0-BASELINE.md: a joint Gaussian random walk over the trailing 300 panel observations
at or before the as-of, drift from the mean step, covariance from the steps across assets, and
cross-horizon covariance min(s_i, s_j) * Sigma so the draws form a path. Drawn at 4,000 samples
instead of M0's 500, which is a free reduction in our own Monte-Carlo error.

Four rounds of candidates — EWMA covariance, Student-t and block-bootstrap innovations, flat and
t-statistic-conditional drift shrinkage, correlation shrinkage, a variance-ratio correction, a
text-derived drift tilt, a text-derived spread widening, and a uniform spread multiplier in both
directions — failed their pre-registered gates. Nine mechanism-level hypotheses are closed. This
is what is left standing, and it is deliberate.

Usage, exactly as the harness invokes it:

    forecast --panels /input/panels --text /input/text --asof YYYY-MM-DD --out /output/forecast.parquet
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import tomllib
import traceback
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

TRAIL = 300
N_DRAWS = 4000
MIN_DRAWS = 200
MAX_DRAWS = 20000
RETURN_TRIPWIRE = 0.2
RATIONALE = "forecast_rationale.md"
META = "forecast_meta.json"
CFG_NAME = "ship"


# ------------------------------------------------------------------ panels and history


def load_panels(panels_dir: Path) -> pd.DataFrame:
    """Every panel row under --panels, falling back to its parent when it holds no parquet."""
    paths = sorted(panels_dir.glob("*.parquet"))
    if not paths:
        paths = sorted(panels_dir.parent.glob("*.parquet"))
    frames = []
    for order, p in enumerate(paths):
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if "asset" not in df.columns and "asset_id" in df.columns:
            df = df.rename(columns={"asset_id": "asset"})
        if not {"date", "asset", "value"} <= set(df.columns):
            continue
        df = df[["date", "asset", "value"]].copy()
        df["_file_order"] = order
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "asset", "value", "_file_order"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"])


def asset_history(panels: pd.DataFrame, asset: str, asof: pd.Timestamp) -> pd.DataFrame:
    """M0 section 3.1: the first panel file holding the asset, rows <= as-of, last 300."""
    rows = panels[panels["asset"] == asset]
    if rows.empty:
        raise KeyError(f"asset {asset!r} appears in no panel")
    rows = rows[rows["_file_order"] == rows["_file_order"].min()]
    rows = rows[rows["date"] <= asof].sort_values("date")
    rows = rows.drop_duplicates(subset="date", keep="last")
    return rows.tail(TRAIL)[["date", "value"]].reset_index(drop=True)


def _gap_limit(dates: pd.Series) -> float:
    """M0 section 3.3: a hole is an interval longer than max(10 x median spacing, 5 days)."""
    spacing = dates.diff().dt.days.dropna()
    if spacing.empty:
        return 5.0
    return max(10.0 * float(spacing.median()), 5.0)


def step_series(hist: pd.DataFrame, target_type: str) -> pd.Series:
    """M0 sections 3.2 and 3.3: steps indexed by the date each step ends on."""
    dates = hist["date"]
    values = hist["value"].astype(float)
    limit = _gap_limit(dates)
    span = dates.diff().dt.days

    if target_type == "log_return":
        if float(np.nanmedian(np.abs(values))) >= RETURN_TRIPWIRE:
            raise ValueError("log_return panel does not look like per-step returns")
        steps = np.log1p(values)
    else:
        steps = values.diff()

    s = pd.Series(steps.to_numpy(), index=dates)
    keep = np.array((span <= limit).to_numpy(), dtype=bool, copy=True)
    keep[0] = False  # row 0 has no preceding interval to test
    return s[keep]


def panel_spacing(hist: pd.DataFrame) -> float:
    dates = hist["date"]
    span = dates.diff().dt.days.dropna()
    good = span[span <= _gap_limit(dates)]
    return float(good.mean()) if not good.empty else float("nan")


def horizon_steps(hist: pd.DataFrame, horizon: int, target_date: pd.Timestamp | None) -> int:
    """M0 section 3.7: convert the declared horizon into panel steps when they disagree by 2x."""
    if len(hist) < 3 or target_date is None or pd.isna(target_date):
        return int(horizon)
    spacing = panel_spacing(hist)
    if not np.isfinite(spacing) or spacing <= 0:
        return int(horizon)
    last = hist["date"].iloc[-1]
    if spacing > 20.0:
        counted = 12 * (target_date.year - last.year) + (target_date.month - last.month)
    else:
        counted = int(round((target_date - last).days / spacing))
    if counted <= 0:
        return int(horizon)
    ratio = max(counted, horizon) / max(min(counted, horizon), 1)
    return int(counted) if ratio >= 2.0 else int(horizon)


# ------------------------------------------------------------------------- the forecast


def _chol(cov: np.ndarray) -> np.ndarray:
    d = cov.shape[0]
    cov = cov + np.eye(d) * 1e-10
    try:
        return np.linalg.cholesky(cov + np.eye(d) * 1e-9)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(np.diag(np.diag(cov + np.eye(d) * 1e-9)))


def draw_primary(
    panels: pd.DataFrame,
    asof: pd.Timestamp,
    assets: list[str],
    horizons: list[int],
    target_type: str,
    unit_id: str,
    n_draws: int,
    step_override: dict[int, int] | None,
) -> tuple[np.ndarray, list[tuple[str, int]], dict]:
    sorted_assets = sorted(assets)
    hist = {a: asset_history(panels, a, asof) for a in sorted_assets}
    frame = pd.DataFrame({a: step_series(hist[a], target_type) for a in sorted_assets})
    frame = frame.dropna(how="any")
    if frame.empty:
        raise ValueError("no date-aligned steps survive the gap rule")
    steps = frame.to_numpy(dtype=float)

    mu = steps.mean(axis=0)
    cov = np.atleast_2d(np.cov(steps, rowvar=False))

    cells = [(a, h) for a in sorted_assets for h in sorted(horizons)]
    a_index = {a: i for i, a in enumerate(sorted_assets)}
    s_of: dict[tuple[str, int], int] = {}
    for asset, horizon in cells:
        if step_override and horizon in step_override:
            s_of[(asset, horizon)] = int(step_override[horizon])
        else:
            s_of[(asset, horizon)] = horizon_steps(hist[asset], horizon, None)
    s_max = max(s_of.values())

    seed = zlib.crc32(f"{unit_id}|{CFG_NAME}".encode()) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)
    k = steps.shape[1]
    z = rng.standard_normal((n_draws, s_max, k))
    cum = np.cumsum(z @ _chol(cov).T, axis=1)

    out = np.empty((n_draws, len(cells)), dtype=float)
    for i, (asset, horizon) in enumerate(cells):
        s = s_of[(asset, horizon)]
        ai = a_index[asset]
        anchor = 0.0 if target_type == "log_return" else float(hist[asset]["value"].iloc[-1])
        out[:, i] = anchor + s * mu[ai] + cum[:, s - 1, ai]

    diag = {
        "seed": seed,
        "n_steps_est": int(steps.shape[0]),
        "steps_per_cell": {f"{a}@{h}": s_of[(a, h)] for a, h in cells},
        "anchor": {
            a: (0.0 if target_type == "log_return" else float(hist[a]["value"].iloc[-1]))
            for a in sorted_assets
        },
        "mu": {a: float(mu[a_index[a]]) for a in sorted_assets},
        "sd": {a: float(np.sqrt(cov[a_index[a], a_index[a]])) for a in sorted_assets},
    }
    return out, cells, diag


def draw_fallback(
    panels: pd.DataFrame,
    asof: pd.Timestamp,
    assets: list[str],
    horizons: list[int],
    target_type: str,
    n_draws: int,
) -> tuple[np.ndarray, list[tuple[str, int]], dict]:
    """Per-asset walk needing no date alignment. Costs the joint term; beats a crash by 3.0."""
    cells = [(a, h) for a in sorted(assets) for h in sorted(horizons)]
    rng = np.random.default_rng(12345)
    out = np.zeros((n_draws, len(cells)), dtype=float)
    for i, (asset, h) in enumerate(cells):
        rows = panels[(panels["asset"] == asset) & (panels["date"] <= asof)]
        v = rows.sort_values("date")["value"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if target_type == "log_return":
            steps = np.log1p(v[-TRAIL:]) if v.size else np.array([0.0])
            anchor = 0.0
        else:
            steps = np.diff(v[-TRAIL:]) if v.size > 1 else np.array([0.0])
            anchor = float(v[-1]) if v.size else 0.0
        mu = float(np.nanmean(steps)) if steps.size else 0.0
        sd = float(np.nanstd(steps, ddof=1)) if steps.size > 1 else 0.0
        if not np.isfinite(sd) or sd <= 0:
            sd = max(abs(anchor) * 0.01, 1e-4)
        out[:, i] = anchor + h * mu + rng.standard_normal(n_draws) * sd * np.sqrt(h)
    return out, cells, {"fallback": "per-asset gaussian rw"}


def draw_last_resort(assets, horizons, n_draws):
    cells = [(a, h) for a in sorted(assets) for h in sorted(horizons)]
    rng = np.random.default_rng(999)
    return rng.standard_normal((n_draws, len(cells))), cells, {
        "fallback": "standard normal; no panel was readable"
    }


# ---------------------------------------------------------------------- card resolution


def find_card(panels_dir: Path) -> Path | None:
    for cand in (panels_dir / "card.toml", panels_dir.parent / "card.toml",
                 Path("/input/card.toml")):
        if cand.exists():
            return cand
    return None


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def resolve_grid(card: dict, spec: dict, panels: pd.DataFrame):
    """Assets, horizons and target type, in the CARD'S OWN ORDER.

    Gate g3 compares the declared grid in forecast_meta.json against the committed one and
    refuses a reordering: a sorted asset_ids list fails with schema_invalid, missing 0 / extra 0,
    which reads like a content error and is purely an ordering one. The parquet is joined on
    (asset, horizon), so the draw order is free; only the sidecar's order is bound.
    """
    tgt = card.get("targets", {}) or {}
    assets = list(tgt.get("asset_ids") or [])
    horizons = [int(h) for h in (tgt.get("horizons") or [])]
    ttype = tgt.get("target_type") or "level"
    if not assets or not horizons:
        stgt = (spec.get("targets") or {}) if isinstance(spec, dict) else {}
        assets = assets or list(stgt.get("asset_ids") or [])
        horizons = horizons or [int(h) for h in (stgt.get("horizons") or [])]
        ttype = stgt.get("target_type") or ttype
    if not assets and not panels.empty:
        assets = sorted(panels["asset"].dropna().unique().tolist())
    if not horizons:
        horizons = [21]
    seen: set[int] = set()
    horizons = [h for h in horizons if not (h in seen or seen.add(h))]
    return assets, horizons, ttype


def monthly_steps(card: dict, spec: dict, assets, horizons, panels: pd.DataFrame):
    """Calendar-month steps for a monthly card, from targets.observation_periods.

    A monthly macro panel stops at the as-of minus its publication lag while the card states a
    business-day horizon; feeding that into a per-step walk forecasts years out. Cross-checked
    against the official qfbench2 helper on all four monthly cards.
    """
    tgt = card.get("targets", {}) or {}
    freq = tgt.get("target_frequency") or card.get("metadata", {}).get("target_frequency")
    if freq != "monthly":
        return None
    periods = ((spec.get("targets") or {}) if isinstance(spec, dict) else {}).get(
        "observation_periods"
    )
    if not periods or len(periods) != len(horizons):
        return None
    out: dict[int, int] = {}
    for asset in assets:
        rows = panels[panels["asset"] == asset]
        if rows.empty:
            return None
        last = rows["date"].max()
        for h, period in zip(horizons, periods):
            try:
                p = pd.Period(str(period), freq="M")
            except Exception:
                return None
            steps = 12 * (p.year - last.year) + (p.month - last.month)
            if steps <= 0:
                return None
            out[h] = int(steps)
    return out


# ------------------------------------------------------------------------------- output


def rationale_text(unit_id, asof, assets, horizons, ttype, n_draws, diag, text_dir, elapsed):
    try:
        docs = sorted(p.name for p in Path(text_dir).glob("*.txt"))
    except Exception:
        docs = []
    steps = diag.get("steps_per_cell", {})
    anchors = diag.get("anchor", {})
    mu = diag.get("mu", {})
    sd = diag.get("sd", {})
    lines = [
        f"# Forecast rationale — {unit_id}",
        "",
        f"As-of {asof}. Target type `{ttype}`. {len(assets)} asset(s) x {len(horizons)} "
        f"horizon(s), {n_draws} joint draws. Wall clock {elapsed:.1f}s.",
        "",
        "## 1. Method, and why this one",
        "",
        "The forecast is the official text-blind baseline's own construction (M0, specified in",
        "the kit's docs/M0-BASELINE.md): a joint Gaussian random walk on the trailing 300 panel",
        "observations at or before the as-of, drift from the mean step, covariance from the steps",
        "across assets, and cross-horizon covariance min(s_i, s_j) * Sigma so the draws form a",
        "path rather than unrelated marginals.",
        "",
        "That is the method on purpose. Four pre-registered rounds were run against a labelled",
        "bench built from the published practice panels: EWMA covariance, Student-t and block",
        "bootstrap innovations, flat and t-statistic-conditional drift shrinkage, correlation",
        "shrinkage, a variance-ratio correction, a text-derived drift tilt, a text-derived spread",
        "widening, and a uniform spread multiplier in both directions. None cleared its gate and",
        "several were significantly worse. So the numerical core stayed at the baseline's own",
        "construction, drawn at more samples than its 500 to cut our own Monte-Carlo error at the",
        "1% and 99% pinball levels.",
        "",
        "## 2. The numbers this run computed",
        "",
    ]
    if diag.get("fallback"):
        lines += [
            f"**Fallback path taken: {diag['fallback']}.** The primary estimator did not complete",
            "on this unit, so the forecast comes from the simpler construction named above.",
            "",
        ]
    lines += ["| cell | panel steps | anchor | drift/step | sd/step |", "|---|---|---|---|---|"]
    for asset in sorted(assets):
        for h in sorted(horizons):
            key = f"{asset}@{h}"
            lines.append(
                f"| {key} | {steps.get(key, h)} | {anchors.get(asset, float('nan')):.6g} | "
                f"{mu.get(asset, float('nan')):.3g} | {sd.get(asset, float('nan')):.3g} |"
            )
    lines += [
        "",
        "Adjustment ledger: anchor = last panel observation at or before the as-of (0 for a",
        "`log_return` target, whose target is a cumulative return); centre = anchor + steps x",
        "drift; spread = the step covariance accumulated over min(s_i, s_j) steps. No other",
        "adjustment was applied — there is nothing else in the ledger, by the decision in 1.",
        "",
        "## 3. What the text corpus contributed",
        "",
        f"Documents present: {len(docs)}" + (f" ({', '.join(docs[:12])})" if docs else ""),
        "",
        "**Nothing, on this run, and that is stated rather than implied.** The corpus was not read",
        "and no term in the ledger above depends on it. A text layer was built and tested on 90",
        "labelled units with their real corpora: a tone-to-drift tilt scored significantly WORSE",
        "than no adjustment, and worse than the same tilt with its feature shuffled across units;",
        "an uncertainty-to-spread widening moved the composite by nothing. Measured directly, the",
        "rank correlation between corpus tone and the standardized forward move was -0.23 — the",
        "wrong sign for the economic story and most likely a regime artifact. Claiming an uplift",
        "the forecast does not contain would misdescribe these draws.",
        "",
        "## 4. What would change this forecast",
        "",
        "A different trailing window, a drift the window could separate from zero, or a",
        "documented regime signal that survived a gate on labelled data from a period other than",
        "the one that produced it. None of those applied here.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(out_path: Path, draws, cells, unit_id, asof, assets, horizons, ttype,
                  diag, text_dir, elapsed):
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    n_draws = int(draws.shape[0])
    frame = pd.DataFrame({
        "draw": np.repeat(np.arange(n_draws, dtype=np.int32), len(cells)).astype("int32"),
        "asset": [c[0] for _ in range(n_draws) for c in cells],
        "horizon": np.tile(np.array([c[1] for c in cells], dtype=np.int32), n_draws).astype("int32"),
        "value": np.ascontiguousarray(draws.reshape(n_draws * len(cells)), dtype=np.float64),
    })
    frame.to_parquet(out_path, index=False)

    (out_dir / META).write_text(json.dumps({
        "unit_id": unit_id,
        "asof": asof,
        "representation": "samples",
        "asset_ids": list(assets),
        "horizons": [int(h) for h in horizons],
        "n_draws": n_draws,
        "target": ttype,
    }, indent=2) + "\n")

    (out_dir / RATIONALE).write_text(
        rationale_text(unit_id, asof, assets, horizons, ttype, n_draws, diag, text_dir, elapsed)
    )


# --------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    t0 = time.time()
    p = argparse.ArgumentParser(prog="forecast")
    p.add_argument("verb", nargs="?", default="forecast")
    p.add_argument("--panels", type=Path, required=True)
    p.add_argument("--text", type=Path, default=Path("/input/text"))
    p.add_argument("--asof", required=True)
    p.add_argument("--out", type=Path, default=Path("/output/forecast.parquet"))
    p.add_argument("--n-draws", type=int, default=N_DRAWS)
    args = p.parse_args(argv)

    if args.verb not in ("forecast", "forecast.py"):
        print(f"unknown verb {args.verb!r}", file=sys.stderr)
        return 2

    assets: list[str] = []
    horizons: list[int] = [21]
    ttype = "level"
    unit_id = "unknown"
    panels = pd.DataFrame(columns=["date", "asset", "value", "_file_order"])
    card: dict = {}
    spec: dict = {}

    try:
        card_path = find_card(args.panels)
        if card_path is not None:
            with card_path.open("rb") as fh:
                card = tomllib.load(fh)
            spec = read_json(card_path.parent / "forecast_spec.json")
        unit_id = str(card.get("task", {}).get("id") or spec.get("card_id") or "unknown")
        panels = load_panels(args.panels)
        assets, horizons, ttype = resolve_grid(card, spec, panels)
    except Exception:
        traceback.print_exc(file=sys.stderr)

    floor = int((card.get("scoring", {}).get("params", {}) or {}).get("n_draws_min", 0) or 0)
    n_draws = int(min(MAX_DRAWS, max(args.n_draws, floor, MIN_DRAWS)))
    asof = pd.Timestamp(args.asof)

    try:
        override = monthly_steps(card, spec, assets, horizons, panels)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        override = None

    draws = cells = None
    diag: dict = {}
    for stage, fn in (
        ("primary", lambda: draw_primary(panels, asof, assets, horizons, ttype, unit_id,
                                         n_draws, override)),
        ("fallback", lambda: draw_fallback(panels, asof, assets, horizons, ttype, n_draws)),
        ("last_resort", lambda: draw_last_resort(assets or ["UNKNOWN"], horizons, n_draws)),
    ):
        try:
            draws, cells, diag = fn()
            if not np.all(np.isfinite(draws)):
                raise ValueError(f"{stage} produced non-finite values")
            break
        except Exception:
            print(f"[forecast] stage {stage} failed:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            draws = cells = None

    if draws is None or cells is None:
        print("[forecast] every stage failed; nothing can be written", file=sys.stderr)
        return 1

    try:
        write_outputs(args.out, draws, cells, unit_id, args.asof, assets, horizons, ttype,
                      diag, args.text, time.time() - t0)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1

    print(f"[forecast] {unit_id}: {len(set(c[0] for c in cells))} asset(s) x "
          f"{len(set(c[1] for c in cells))} horizon(s), {n_draws} draws, "
          f"{time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
