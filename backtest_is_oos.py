#!/usr/bin/env python
"""IS / OOS backtester for Algothon 2026.

Mirrors eval.py economics (fees, limits, integer clip, mark-only last day)
but reports mu / sigma / Sharpe / score on separate windows without the
per-day print spam.

Default split (edit below or pass CLI flags):
  IS:  days 50  -> 160   (tune / sanity-check here)
  OOS: days 160 -> 500   (official eval window)

Also reports the last-250-day slice (leaderboard-style) and 50-day blocks.

Usage:
  python backtest_is_oos.py
  python backtest_is_oos.py --strategy larpSharpe
  python backtest_is_oos.py --is-start 40 --is-end 160 --oos-start 160 --oos-end 500
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import matplotlib

if os.environ.get("MPLBACKEND", "").lower() in ("", "agg"):
    for backend in ("TkAgg", "Qt5Agg", "WXAgg"):
        try:
            matplotlib.use(backend, force=True)
            break
        except ImportError:
            continue

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- defaults (match eval.py economics) ---
DEFAULT_STRATEGY = "strategy"
PRICES_FILE = os.path.join(_SCRIPT_DIR, "prices.txt")

DEFAULT_COMM_RATE = 0.0001
INST0_COMM_RATE = 0.00002
DEFAULT_DLR_LIMIT = 10_000
INST0_DLR_LIMIT = 100_000
SCORE_PARAM = 1.0
BLOCK_SIZE = 50
MIN_VOLUME_FLAG = 25_000  # Testing Round inactivity threshold


def score(mu: float, sigma: float, param: float = SCORE_PARAM) -> float:
    if mu <= 0 or sigma < 1e-10:
        return float(mu)
    sr = np.sqrt(250.0) * mu / sigma
    return float(mu * (sr**2 / (sr**2 + param**2)))


def load_prices(fn: str) -> np.ndarray:
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return df.values.T  # (nInst, nt)


def load_strategy(module_name: str):
    sys.path.insert(0, _SCRIPT_DIR)
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "getMyPosition"):
        raise AttributeError(f"{module_name}.py must define getMyPosition(prcSoFar)")
    return mod.getMyPosition


def calc_pl(prc_hist: np.ndarray, get_position, start_day: int, end_day: int):
    """Replay eval.py day loop over [start_day, end_day].

    Positions are first requested on day (start_day - 1). Day end_day is
    mark-only (no new trades), matching official eval.py.
    """
    n_inst = prc_hist.shape[0]
    comm_rate = np.full(n_inst, DEFAULT_COMM_RATE)
    comm_rate[0] = INST0_COMM_RATE
    dlr_limit = np.full(n_inst, DEFAULT_DLR_LIMIT, dtype=float)
    dlr_limit[0] = INST0_DLR_LIMIT

    cash = 0.0
    cur_pos = np.zeros(n_inst)
    tot_dvolume = 0.0
    value = 0.0
    comm = 0.0

    days = []
    pll = []
    dvols = []  # daily dollar volume traded (for turnover diagnostics)

    loop_start = start_day - 1
    for t in range(loop_start, end_day + 1):
        prc_so_far = prc_hist[:, :t]
        cur_prices = prc_so_far[:, -1]

        if t < end_day:
            new_pos_orig = get_position(prc_so_far)
            pos_limits = (dlr_limit / cur_prices).astype(int)
            new_pos = np.clip(new_pos_orig, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = np.array(cur_pos)

        delta_pos = new_pos - cur_pos
        cash -= cur_prices.dot(delta_pos) + comm

        dvolumes = cur_prices * np.abs(delta_pos)
        dvolume = float(np.sum(dvolumes))
        tot_dvolume += dvolume
        comm = float(np.sum(dvolumes * comm_rate))

        cur_pos = np.array(new_pos)
        pos_value = float(cur_pos.dot(cur_prices))
        today_pl = cash + pos_value - value
        value = cash + pos_value

        if t >= start_day:
            days.append(t)
            pll.append(today_pl)
            dvols.append(dvolume)

    days = np.asarray(days, dtype=int)
    pll = np.asarray(pll, dtype=float)
    dvols = np.asarray(dvols, dtype=float)
    ret = (value / tot_dvolume) if tot_dvolume > 0 else 0.0
    return days, pll, dvols, tot_dvolume, value, ret


def summarise(pll: np.ndarray, dvols: np.ndarray | None = None) -> dict:
    n = len(pll)
    if n == 0:
        return {
            "n": 0,
            "mu": 0.0,
            "sigma": 0.0,
            "sharpe": 0.0,
            "score": 0.0,
            "total_pl": 0.0,
            "hit_rate": 0.0,
            "max_dd": 0.0,
            "avg_daily_dvol": 0.0,
            "tot_dvol": 0.0,
        }

    mu = float(np.mean(pll))
    # population std to match eval.py's np.std(pll) default (ddof=0)
    sigma = float(np.std(pll))
    sharpe = (np.sqrt(250.0) * mu / sigma) if sigma > 0 else 0.0
    cum = np.cumsum(pll)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum - peak)) if n else 0.0
    tot_dvol = float(np.sum(dvols)) if dvols is not None else 0.0
    avg_dvol = float(np.mean(dvols)) if dvols is not None and len(dvols) else 0.0

    return {
        "n": n,
        "mu": mu,
        "sigma": sigma,
        "sharpe": float(sharpe),
        "score": score(mu, sigma),
        "total_pl": float(np.sum(pll)),
        "hit_rate": float(np.mean(pll > 0)),
        "max_dd": max_dd,
        "avg_daily_dvol": avg_dvol,
        "tot_dvol": tot_dvol,
    }


def slice_mask(days: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Inclusive day range on scored days (same convention as eval)."""
    return (days >= lo) & (days <= hi)


def block_table(
    days: np.ndarray,
    pll: np.ndarray,
    dvols: np.ndarray,
    start_day: int,
    block: int = BLOCK_SIZE,
):
    rows = []
    if len(days) == 0:
        return rows
    blocks = (days - start_day) // block
    for b in np.unique(blocks):
        m = blocks == b
        s = summarise(pll[m], dvols[m])
        d0 = int(days[m][0])
        d1 = int(days[m][-1])
        rows.append((f"{d0}-{d1}", s))
    return rows


def fmt_row(label: str, s: dict) -> str:
    vol_flag = ""
    if s["tot_dvol"] > 0 and s["tot_dvol"] < MIN_VOLUME_FLAG:
        vol_flag = "  !! under $25k vol"
    return (
        f"{label:<22}  n={s['n']:>3}  "
        f"mu={s['mu']:>8.2f}  sigma={s['sigma']:>8.2f}  "
        f"Sharpe={s['sharpe']:>6.2f}  score={s['score']:>8.2f}  "
        f"sumPL={s['total_pl']:>10.1f}  hit={s['hit_rate']:>5.1%}  "
        f"maxDD={s['max_dd']:>9.1f}  dVol={s['tot_dvol']:>10.0f}"
        f"{vol_flag}"
    )


def plot_windows(days, pll, is_start, is_end, oos_start, oos_end, out_file):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # cumulative PnL with IS/OOS shading
    ax = axes[0]
    cum = np.cumsum(pll)
    ax.plot(days, cum, color="#1f77b4", lw=1.8, label="Cumulative PnL")
    ax.axvspan(is_start, is_end, color="#2ca02c", alpha=0.12, label=f"IS {is_start}-{is_end}")
    ax.axvspan(oos_start, oos_end, color="#ff7f0e", alpha=0.12, label=f"OOS {oos_start}-{oos_end}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("IS / OOS cumulative PnL")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)

    # daily bars
    ax = axes[1]
    colors = np.where(pll >= 0, "#2ca02c", "#d62728")
    ax.bar(days, pll, width=0.8, color=colors, alpha=0.85)
    ax.axvline(oos_start, color="#ff7f0e", ls="--", lw=1.5, label=f"OOS start {oos_start}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Daily PnL ($)")
    ax.set_title("Daily PnL")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    print(f"Saved plot -> {out_file}")
    return fig


def parse_args():
    p = argparse.ArgumentParser(description="Algothon IS/OOS backtester")
    p.add_argument("--strategy", default=DEFAULT_STRATEGY,
                   help="module name in this folder with getMyPosition (default: larpSharpe)")
    p.add_argument("--prices", default=PRICES_FILE, help="path to prices.txt")
    p.add_argument("--is-start", type=int, default=50)
    p.add_argument("--is-end", type=int, default=160)
    p.add_argument("--oos-start", type=int, default=160)
    p.add_argument("--oos-end", type=int, default=750)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--quiet-strategy-path", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    get_position = load_strategy(args.strategy)
    prc = load_prices(args.prices)
    n_inst, nt = prc.shape

    is_start, is_end = args.is_start, args.is_end
    oos_start, oos_end = args.oos_start, args.oos_end

    if not (1 <= is_start < is_end <= oos_start <= oos_end <= nt):
        raise SystemExit(
            f"Bad windows vs prices length {nt}: "
            f"IS[{is_start},{is_end}] OOS[{oos_start},{oos_end}]"
        )

    # One continuous replay from IS start through OOS end so inventory
    # carries into the official window the way live trading would.
    days, pll, dvols, tot_dvol, value, ret = calc_pl(
        prc, get_position, start_day=is_start, end_day=oos_end
    )

    print("=" * 100)
    print(f"Strategy: {args.strategy}.getMyPosition")
    print(f"Prices:   {args.prices}  ({n_inst} inst x {nt} days)")
    print(f"Replay:   days {is_start} -> {oos_end}  (continuous; mark-only on {oos_end})")
    print(f"Final value: {value:.2f}   value/totDVol: {ret:.5f}   totDVol: {tot_dvol:.0f}")
    print("=" * 100)

    windows = [
        ("FULL (IS+OOS)", is_start, oos_end),
        ("IS", is_start, is_end),
        ("OOS (official)", oos_start, oos_end),
        ("LAST 250 (lb-style)", max(oos_end - 249, oos_start), oos_end),
    ]

    print("\n-- Window summary --")
    print(
        f"{'window':<22}  {'n':>5}  {'mu':>8}  {'sigma':>8}  "
        f"{'Sharpe':>6}  {'score':>8}  {'sumPL':>10}  {'hit':>5}  "
        f"{'maxDD':>9}  {'dVol':>10}"
    )
    for label, lo, hi in windows:
        m = slice_mask(days, lo, hi)
        s = summarise(pll[m], dvols[m])
        # relabel with actual day span
        label2 = f"{label} {lo}-{hi}"
        print(fmt_row(label2, s))

    # Official-eval apples-to-apples: fresh run starting at day 160 only
    # (positions start flat at OOS open — matches eval.py exactly).
    days_o, pll_o, dvols_o, tot_o, val_o, ret_o = calc_pl(
        prc, get_position, start_day=oos_start, end_day=oos_end
    )
    s_official = summarise(pll_o, dvols_o)
    print("\n-- Official eval.py replica (flat start at OOS, no IS carry) --")
    print(fmt_row(f"EVAL {oos_start}-{oos_end}", s_official))
    print(f"  final value={val_o:.2f}  return={ret_o:.5f}  totDVol={tot_o:.0f}")

    print(f"\n-- {BLOCK_SIZE}-day OOS blocks (continuous-run OOS slice) --")
    m_oos = slice_mask(days, oos_start, oos_end)
    for label, s in block_table(
        days[m_oos], pll[m_oos], dvols[m_oos], oos_start, BLOCK_SIZE
    ):
        print(fmt_row(f"block {label}", s))

    # Overfit sniff test
    s_is = summarise(pll[slice_mask(days, is_start, is_end)], dvols[slice_mask(days, is_start, is_end)])
    s_oos = summarise(pll[slice_mask(days, oos_start, oos_end)], dvols[slice_mask(days, oos_start, oos_end)])
    print("\n-- Quick read --")
    if s_is["score"] > 0 and s_oos["score"] <= 0:
        print("WARN: positive IS score, non-positive OOS -> likely overfit / regime break.")
    elif s_is["mu"] > 0 and s_oos["mu"] > 0 and s_oos["sharpe"] < 0.5 * s_is["sharpe"]:
        print("WARN: OOS Sharpe << IS Sharpe -> edge may be fragile; check turnover / params.")
    elif s_oos["score"] > 0 and s_is["score"] > 0:
        print("OK: both IS and OOS scores positive (still check block stability).")
    else:
        print("INFO: inspect mu/sigma/score rows above; at least one window is weak.")

    if s_official["tot_dvol"] < MIN_VOLUME_FLAG:
        print(f"WARN: official-window dollar volume {s_official['tot_dvol']:.0f} < {MIN_VOLUME_FLAG}.")

    out_png = os.path.join(_SCRIPT_DIR, "backtest_is_oos.png")
    if not args.no_plot:
        fig = plot_windows(days, pll, is_start, is_end, oos_start, oos_end, out_png)
        backend = plt.get_backend().lower()
        if any(backend.startswith(b) for b in ("tkagg", "qt5agg", "qt4agg", "wxagg", "macosx")):
            print("Displaying plot (close window to finish)...")
            plt.show(block=True)
        else:
            if sys.platform == "win32" and os.path.exists(out_png):
                os.startfile(os.path.abspath(out_png))
        plt.close(fig)


if __name__ == "__main__":
    main()
