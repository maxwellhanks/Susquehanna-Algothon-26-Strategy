#!/usr/bin/env python
"""IS / OOS backtester for Algothon 2026.

Mirrors eval.py economics (fees, limits, integer clip, mark-only last day)
but reports mu / sigma / Sharpe / score on separate windows without the
per-day print spam.

Default split (edit below or pass CLI flags):
  IS:  days 50  -> 160    (tune / sanity-check here)
  OOS: days 160 -> 1000   (official eval window)

Also reports the last-250-day slice (leaderboard-style) and 50-day blocks.

Usage:
  python backtest_is_oos.py
  python backtest_is_oos.py --strategy larpSharpe
  python backtest_is_oos.py --is-start 40 --is-end 160 --oos-start 160 --oos-end 1000
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
from typing import NamedTuple

import importlib.util as _ilu

import matplotlib

# Pick an interactive backend only if its GUI toolkit is actually importable
# (matplotlib.use() alone doesn't verify this until plot time). Honour an
# explicit MPLBACKEND; otherwise fall back to Agg (file output only).
if not os.environ.get("MPLBACKEND"):
    for _backend, _toolkit in (("TkAgg", "tkinter"),
                               ("Qt5Agg", "PyQt5"),
                               ("WXAgg", "wx"),
                               ("MacOSX", "objc")):
        if _ilu.find_spec(_toolkit) is not None:
            try:
                matplotlib.use(_backend, force=False)
                break
            except Exception:
                continue
    else:
        matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- defaults (match eval.py economics) ---
DEFAULT_STRATEGY = "strategy v8"
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


def reset_strategy_state(get_position) -> None:
    """Reset a stateful strategy before an independent replay when supported.

    Without this, the second ("fresh official") replay can retain fits cached
    at the end of the first replay.  Rewinding from day 1000 to day 159 then
    leaks future-fitted coefficients until the cache happens to refit.
    """
    module = sys.modules.get(get_position.__module__)
    reset = getattr(module, "reset_state", None) if module is not None else None
    if callable(reset):
        reset()


class Replay(NamedTuple):
    """Everything one replay produces.  Attribute access, so adding a field
    later can't silently break a positional unpack at a call site."""
    days: np.ndarray        # (T,)      scored day indices
    pll: np.ndarray         # (T,)      portfolio daily PnL
    dvols: np.ndarray       # (T,)      portfolio daily dollar volume
    tot_dvol: float
    value: float
    ret: float
    inst_pll: np.ndarray    # (T, N)    per-inst daily NET PnL (cash basis)
    util: np.ndarray        # (T, N)    |pos*p| / dollar limit
    inst_dvol: np.ndarray   # (T, N)    per-inst dollar volume traded on day t
    inst_fee: np.ndarray    # (T, N)    fee ACCRUED on day t's trades
    inst_fee_paid: np.ndarray  # (T, N) fee CHARGED on day t (= accrued t-1)
    comm_rate: np.ndarray   # (N,)      per-inst commission rate


def calc_pl(prc_hist: np.ndarray, get_position, start_day: int, end_day: int) -> Replay:
    """Replay eval.py day loop over [start_day, end_day].

    Positions are first requested on day (start_day - 1). Day end_day is
    mark-only (no new trades), matching official eval.py.

    Also records, on scored days only:
      inst_pll[t, i]      = pos_i(t-1)*(p_i(t) - p_i(t-1)) - comm_i(paid at t)
                            (sums across i to the day's total PnL exactly, since
                             today_pl = pos_old . dP - comm_prev in eval mechanics)
      util[t, i]          = |pos_i(t) * p_i(t)| / dlr_limit_i   (post-clip)
      inst_dvol[t, i]     = |delta_pos_i(t)| * p_i(t)           (traded at t)
      inst_fee[t, i]      = inst_dvol[t, i] * comm_rate_i       (accrued at t)
      inst_fee_paid[t, i] = inst_fee[t-1, i]                    (charged at t)

    Fee timing note: eval charges day t's commission against day t+1's cash,
    so `inst_fee` (accrual, aligned with the volume that generated it) is the
    correct numerator for any fee-per-dollar-traded ratio, while
    `inst_fee_paid` is what `inst_pll` actually deducts.  They differ only by
    a one-day shift at the window edges.
    """
    reset_strategy_state(get_position)
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
    comm_vec = np.zeros(n_inst)      # per-inst commission carried to next day
    prev_prices = None

    days = []
    pll = []
    dvols = []  # daily dollar volume traded (for turnover diagnostics)
    inst_pll = []   # per-instrument daily PnL rows
    util = []       # per-instrument |pos*price|/limit rows
    inst_dvol = []  # per-instrument dollar volume traded rows
    inst_fee = []       # per-instrument fee accrued on today's trades
    inst_fee_paid = []  # per-instrument fee actually charged today

    loop_start = start_day - 1
    for t in range(loop_start, end_day + 1):
        prc_so_far = prc_hist[:, :t]
        cur_prices = prc_so_far[:, -1]

        # fee charged against today's cash = yesterday's accrual
        fee_paid_today = comm_vec.copy()

        # per-instrument PnL for *this* day, before rebalancing
        if prev_prices is not None:
            day_inst_pl = cur_pos * (cur_prices - prev_prices) - comm_vec
        else:
            day_inst_pl = np.zeros(n_inst)

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
        comm_vec = dvolumes * comm_rate
        comm = float(np.sum(comm_vec))

        cur_pos = np.array(new_pos)
        pos_value = float(cur_pos.dot(cur_prices))
        today_pl = cash + pos_value - value
        value = cash + pos_value

        if t >= start_day:
            days.append(t)
            pll.append(today_pl)
            dvols.append(dvolume)
            inst_pll.append(day_inst_pl)
            util.append(np.abs(cur_pos * cur_prices) / dlr_limit)
            inst_dvol.append(dvolumes)
            inst_fee.append(comm_vec.copy())
            inst_fee_paid.append(fee_paid_today)

        prev_prices = cur_prices

    days = np.asarray(days, dtype=int)
    pll = np.asarray(pll, dtype=float)
    dvols = np.asarray(dvols, dtype=float)
    inst_pll = np.asarray(inst_pll, dtype=float)   # (n_days, n_inst)
    util = np.asarray(util, dtype=float)           # (n_days, n_inst)
    inst_dvol = np.asarray(inst_dvol, dtype=float)
    inst_fee = np.asarray(inst_fee, dtype=float)
    inst_fee_paid = np.asarray(inst_fee_paid, dtype=float)
    ret = (value / tot_dvolume) if tot_dvolume > 0 else 0.0
    return Replay(days, pll, dvols, tot_dvolume, value, ret, inst_pll, util,
                  inst_dvol, inst_fee, inst_fee_paid, comm_rate)


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




def plot_instrument_pnl(days, inst_pll, out_file, top_n=10):
    """Top-N cumulative per-instrument PnL lines + total PnL bar for all 51.

    Top-N by |total PnL| keeps the line chart readable; the bar panel below
    still shows every instrument so nothing is hidden.
    """
    n_inst = inst_pll.shape[1]
    totals = inst_pll.sum(axis=0)
    order = np.argsort(-np.abs(totals))
    top = order[:top_n]

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    ax = axes[0]
    cum = np.cumsum(inst_pll, axis=0)
    cmap = plt.get_cmap("tab10")
    for rank, k in enumerate(top):
        lbl = f"inst {k}" + (" (ALGO)" if k == 0 else "") + f"  {totals[k]:+,.0f}"
        ax.plot(days, cum[:, k], lw=1.6, color=cmap(rank % 10), label=lbl)
    rest = np.setdiff1d(np.arange(n_inst), top)
    if len(rest):
        ax.plot(days, cum[:, rest].sum(axis=1), lw=1.2, ls="--", color="grey",
                label=f"other {len(rest)} combined  {totals[rest].sum():+,.0f}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title(f"Per-instrument cumulative PnL — top {top_n} by |total|")
    ax.legend(loc="best", fontsize=8, ncols=2)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    colors = np.where(totals >= 0, "#2ca02c", "#d62728")
    ax.bar(np.arange(n_inst), totals, color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Instrument index (0 = ALGO)")
    ax.set_ylabel("Total PnL ($)")
    ax.set_title("Total PnL by instrument (all 51)")
    ax.set_xticks(np.arange(0, n_inst, 2))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    print(f"Saved plot -> {out_file}")
    return fig


def plot_limit_usage(days, util, out_file):
    """How much of the per-instrument dollar limits the book actually uses.

    Panel 1: portfolio-level capacity usage through time, split into
             non-ALGO (sum |pos*p| / 50*$10k) and ALGO (|pos*p| / $100k),
             since ALGO's limit dominates raw capacity.
    Panel 2: per-instrument mean and max daily utilization of its own limit.
    """
    n_inst = util.shape[1]
    non_algo = util[:, 1:].mean(axis=1)          # mean fraction of $10k per inst
    algo = util[:, 0]                            # fraction of $100k

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    ax = axes[0]
    ax.plot(days, 100 * non_algo, lw=1.6, color="#1f77b4",
            label="non-ALGO: mean % of $10k limits in use")
    ax.plot(days, 100 * algo, lw=1.6, color="#ff7f0e",
            label="ALGO: % of $100k limit in use")
    ax.axhline(100, color="black", ls=":", lw=1.0)
    ax.set_ylabel("% of dollar limit")
    ax.set_ylim(0, 105)
    ax.set_title("Daily limit utilization through time")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    mean_u = 100 * util.mean(axis=0)
    max_u = 100 * util.max(axis=0)
    xs = np.arange(n_inst)
    ax.bar(xs, max_u, color="#c6dbef", label="max day")
    ax.bar(xs, mean_u, color="#1f77b4", label="mean day")
    ax.axhline(100, color="black", ls=":", lw=1.0)
    ax.set_xlabel("Instrument index (0 = ALGO)")
    ax.set_ylabel("% of own dollar limit")
    ax.set_title("Per-instrument limit utilization (mean vs max)")
    ax.set_xticks(np.arange(0, n_inst, 2))
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    print(f"Saved plot -> {out_file}")
    return fig


# ---------------------------------------------------------------------------
# Small-multiple grid: daily (NOT cumulative) PnL for every instrument
# ---------------------------------------------------------------------------

POS_C = "#2ca02c"
NEG_C = "#d62728"


def _grid_axes(n_panels, ncols, panel_w=2.6, panel_h=1.55, sharex=True, sharey=False):
    """Build a ceil(n/ncols) x ncols grid and blank the unused cells."""
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_w * ncols, panel_h * nrows),
        sharex=sharex, sharey=sharey,
        squeeze=False,
    )
    flat = axes.ravel()
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    return fig, flat, nrows


def _rolling_sum(a, w):
    """Trailing rolling sum along axis 0; first w-1 rows are NaN."""
    a = np.asarray(a, dtype=float)
    c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)], axis=0)
    out = np.full(a.shape, np.nan)
    if a.shape[0] >= w:
        out[w - 1:] = c[w:] - c[:-w]
    return out


def plot_daily_pnl_grid(days, inst_pll, out_file, ncols=6, oos_start=None,
                        roll=21, share_y=False):
    """One panel per instrument: DAILY net PnL (not cumulative).

    51 instruments -> 9 rows x 6 cols (the last row holds 3).  Bars are the
    raw daily PnL; the black line is a `roll`-day rolling mean of the same
    quantity, so drift is visible without switching to a cumulative view.

    share_y=False lets each instrument autoscale (small books stay readable);
    share_y=True makes magnitudes comparable across panels.
    """
    n_inst = inst_pll.shape[1]
    totals = inst_pll.sum(axis=0)
    sds = inst_pll.std(axis=0)

    fig, axes, nrows = _grid_axes(n_inst, ncols, sharey=share_y)
    roll_mean = _rolling_sum(inst_pll, roll) / roll if roll and roll > 1 else None

    for i in range(n_inst):
        ax = axes[i]
        y = inst_pll[:, i]
        ax.vlines(days, 0, y, colors=np.where(y >= 0, POS_C, NEG_C),
                  lw=0.6, alpha=0.85)
        if roll_mean is not None:
            ax.plot(days, roll_mean[:, i], color="black", lw=0.9, alpha=0.8)
        ax.axhline(0, color="black", lw=0.6)
        if oos_start is not None:
            ax.axvline(oos_start, color="#ff7f0e", ls="--", lw=0.9, alpha=0.9)

        name = "ALGO" if i == 0 else f"#{i}"
        ax.set_title(f"{name}   Σ{totals[i]:+,.0f}   σ{sds[i]:,.0f}",
                     fontsize=7, pad=2,
                     color=("#1a6b1a" if totals[i] >= 0 else "#9b1c1c"))
        ax.tick_params(labelsize=6, length=2, pad=1)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # y-label only on the left column, x-label only on the bottom row
    for i in range(n_inst):
        if i % ncols == 0:
            axes[i].set_ylabel("$/day", fontsize=7)
        if i >= n_inst - ncols:
            axes[i].set_xlabel("day", fontsize=7)

    sub = f"black line = {roll}d rolling mean" if roll and roll > 1 else ""
    if oos_start is not None:
        sub += ("; " if sub else "") + f"dashed orange = OOS start (day {oos_start})"
    fig.suptitle(f"Daily per-instrument PnL (not cumulative) — {n_inst} instruments"
                 + (f"\n{sub}" if sub else ""), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    print(f"Saved plot -> {out_file}")
    fig.savefig(out_file, dpi=140)
    return fig


# ---------------------------------------------------------------------------
# Fee / traded-volume economics
# ---------------------------------------------------------------------------

def cost_table(rep: Replay, mask: np.ndarray | None = None) -> dict:
    """Per-instrument cost economics over the masked days.

    fee_bps  = 1e4 * fee / volume   -> IDENTICALLY the commission rate, since
               fee := volume * rate.  Kept as a build check, not a signal.
    gross_bps= 1e4 * gross PnL / volume  -> edge earned per dollar traded.
    net_bps  = gross_bps - fee_bps       -> what survives the commission.

    An instrument is only worth trading if gross_bps clears its fee_bps
    hurdle (1 bp everywhere, 0.2 bp on ALGO).
    """
    m = slice(None) if mask is None else mask
    vol = rep.inst_dvol[m].sum(axis=0)
    fee = rep.inst_fee[m].sum(axis=0)
    net = rep.inst_pll[m].sum(axis=0)
    fee_paid = rep.inst_fee_paid[m].sum(axis=0)
    gross = net + fee_paid

    with np.errstate(divide="ignore", invalid="ignore"):
        fee_bps = np.where(vol > 0, 1e4 * fee / vol, np.nan)
        gross_bps = np.where(vol > 0, 1e4 * gross / vol, np.nan)
        net_bps = np.where(vol > 0, 1e4 * net / vol, np.nan)
        fee_share = np.where(gross > 0, fee / gross, np.nan)

    return {
        "vol": vol, "fee": fee, "net": net, "gross": gross,
        "fee_bps": fee_bps, "gross_bps": gross_bps, "net_bps": net_bps,
        "fee_share_of_gross": fee_share,
        "trade_days": (rep.inst_dvol[m] > 0).sum(axis=0),
        "rate_bps": 1e4 * rep.comm_rate,
        "fee_paid_total": float(fee_paid.sum()),
    }


def print_cost_table(ct: dict, label: str, top_n: int = 15) -> None:
    n = len(ct["vol"])
    order = np.argsort(-ct["vol"])
    print(f"\n-- Per-instrument cost economics [{label}] "
          f"(worst-to-best by volume, top {top_n}) --")
    print(f"{'inst':>5} {'dVol$':>12} {'fee$':>9} {'gross$':>10} {'net$':>10} "
          f"{'gross_bp':>9} {'fee_bp':>7} {'net_bp':>8} {'fee/gross':>10} {'tdays':>6}")
    for i in order[:top_n]:
        fs = ct["fee_share_of_gross"][i]
        print(f"{i:>5} {ct['vol'][i]:>12,.0f} {ct['fee'][i]:>9,.1f} "
              f"{ct['gross'][i]:>10,.0f} {ct['net'][i]:>10,.0f} "
              f"{ct['gross_bps'][i]:>9.2f} {ct['fee_bps'][i]:>7.2f} "
              f"{ct['net_bps'][i]:>8.2f} "
              f"{(f'{fs:>9.1%}' if np.isfinite(fs) else '        --')} "
              f"{ct['trade_days'][i]:>6}")
    tot_v, tot_f, tot_g = ct["vol"].sum(), ct["fee"].sum(), ct["gross"].sum()
    print(f"{'ALL':>5} {tot_v:>12,.0f} {tot_f:>9,.1f} {tot_g:>10,.0f} "
          f"{ct['net'].sum():>10,.0f} {1e4*tot_g/tot_v:>9.2f} "
          f"{1e4*tot_f/tot_v:>7.2f} {1e4*(tot_g-tot_f)/tot_v:>8.2f} "
          f"{tot_f/tot_g if tot_g > 0 else float('nan'):>9.1%} "
          f"{'':>6}")
    dead = np.where(ct["vol"] <= 0)[0]
    if len(dead):
        print(f"  never traded ({len(dead)}/{n}): {list(map(int, dead))}")
    paid = ct["fee_paid_total"]
    if abs(paid - tot_f) > 1e-9:
        print(f"  note: fees ACCRUED in window ${tot_f:,.2f} vs fees CHARGED "
              f"${paid:,.2f} (${paid - tot_f:+,.2f}); the gap is the one-day "
              f"billing lag at the window edges.  gross := net + charged, "
              f"so gross_bp - fee_bp differs from net_bp by ~{1e4*(paid-tot_f)/tot_v:+.4f} bp.")


def plot_fee_bars(ct: dict, out_file, label: str = ""):
    """All-instrument bar view of the fee/volume economics."""
    n_inst = len(ct["vol"])
    xs = np.arange(n_inst)
    fig, axes = plt.subplots(4, 1, figsize=(14, 13.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.0, 0.6, 1.5]})

    ax = axes[0]
    ax.bar(xs, ct["vol"], color="#4C72B0", alpha=0.9)
    ax.set_ylabel("Dollar volume traded ($)")
    ax.set_title("Total traded dollar volume per instrument")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(xs, ct["fee"], color="#DD8452", alpha=0.9)
    ax.set_ylabel("Commission paid ($)")
    ax.set_title("Total fees per instrument")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    ax.bar(xs, ct["fee_bps"], color="#8172B3", alpha=0.9)
    for r in np.unique(ct["rate_bps"]):
        ax.axhline(r, color="black", ls=":", lw=1.0)
    ax.set_ylabel("fee / volume (bps)")
    ax.set_ylim(0, max(1.4 * np.nanmax(ct["rate_bps"]), 0.1))
    ax.set_title("fee / traded volume — flat by construction "
                 "(fee := volume x rate: 1.0 bp, ALGO 0.2 bp). Build check only.")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[3]
    g, f, nb = ct["gross_bps"], ct["fee_bps"], ct["net_bps"]
    ax.bar(xs, nb, width=0.7, color=np.where(np.nan_to_num(nb) >= 0, POS_C, NEG_C),
           alpha=0.9, label="net PnL / volume (what you keep)")
    ax.plot(xs, g, "_", ms=9, mew=1.6, color="#4C72B0",
            label="gross PnL / volume (pre-fee)")
    ax.plot(xs, f, ls="--", lw=1.3, color="#DD8452",
            label="fee / volume (break-even hurdle)")
    ax.plot(xs, -f, ls=":", lw=1.0, color="#DD8452")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Instrument index (0 = ALGO)")
    ax.set_ylabel("bps per dollar traded")
    ax.set_title("Edge vs cost per dollar traded — the panel that matters. "
                 "A red bar means the instrument does not earn back its own commission; "
                 "the gap between the blue dash and the bar top IS the fee.")
    ax.set_xticks(np.arange(0, n_inst, 2))
    ax.legend(loc="best", fontsize=8, ncols=3)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Fee / traded-volume economics{'  —  ' + label if label else ''}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_file, dpi=140)
    return fig


def plot_cost_series_single(rep: Replay, inst_ind: int, out_file,
                            window: int = 63, oos_start=None):
    """Continuous (through-time) fee/volume view for ONE instrument."""
    days = rep.days
    vol = rep.inst_dvol[:, inst_ind]
    fee = rep.inst_fee[:, inst_ind]
    net = rep.inst_pll[:, inst_ind]
    gross = net + rep.inst_fee_paid[:, inst_ind]
    rate_bps = 1e4 * rep.comm_rate[inst_ind]

    rv = _rolling_sum(vol, window)
    rf = _rolling_sum(fee, window)
    rn = _rolling_sum(net, window)
    rg = _rolling_sum(gross, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        fee_bps = np.where(rv > 0, 1e4 * rf / rv, np.nan)
        net_bps = np.where(rv > 0, 1e4 * rn / rv, np.nan)
        gross_bps = np.where(rv > 0, 1e4 * rg / rv, np.nan)

    name = "ALGO (inst 0)" if inst_ind == 0 else f"instrument {inst_ind}"
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    ax = axes[0]
    ax.plot(days, np.cumsum(vol), color="#4C72B0", lw=1.8, label="cumulative volume ($)")
    ax.set_ylabel("Cumulative volume ($)", color="#4C72B0")
    ax.tick_params(axis="y", labelcolor="#4C72B0")
    ax2 = ax.twinx()
    ax2.plot(days, np.cumsum(fee), color="#DD8452", lw=1.8, label="cumulative fees ($)")
    ax2.set_ylabel("Cumulative fees ($)", color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")
    ax.set_title(f"{name} — traded volume and fees accumulate at a fixed ratio "
                 f"({rate_bps:.2f} bp)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.plot(days, gross_bps, color="#4C72B0", lw=1.5, label=f"gross PnL / volume ({window}d roll)")
    ax.plot(days, net_bps, color="black", lw=1.6, label=f"net PnL / volume ({window}d roll)")
    ax.plot(days, fee_bps, color="#DD8452", lw=1.6, ls="--",
            label=f"fee / volume = {rate_bps:.2f} bp (constant)")
    ax.fill_between(days, 0, net_bps, where=np.nan_to_num(net_bps) >= 0,
                    color=POS_C, alpha=0.18, interpolate=True)
    ax.fill_between(days, 0, net_bps, where=np.nan_to_num(net_bps) < 0,
                    color=NEG_C, alpha=0.18, interpolate=True)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("bps per dollar traded")
    ax.set_title("Rolling edge per dollar traded vs the commission hurdle")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    ax.plot(days, np.cumsum(gross), color="#4C72B0", lw=1.6, label="cumulative gross PnL")
    ax.plot(days, np.cumsum(net), color="black", lw=1.8, label="cumulative net PnL")
    ax.plot(days, -np.cumsum(fee), color="#DD8452", lw=1.6, label="cumulative fees (negated)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("$")
    ax.set_title("Where the money went")
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    for ax in axes:
        if oos_start is not None:
            ax.axvline(oos_start, color="#ff7f0e", ls="--", lw=1.2, alpha=0.9)

    fig.tight_layout()
    fig.savefig(out_file, dpi=140)
    return fig


def plot_cost_series_grid(rep: Replay, out_file, ncols=6, window=63, oos_start=None):
    """Same continuous view as `plot_cost_series_single`, all instruments.

    Per panel: rolling net PnL per dollar traded (black) against that
    instrument's constant fee-per-dollar hurdle (orange dashed).  Green fill
    = clearing costs, red fill = trading at a loss net of commission.
    """
    n_inst = rep.inst_dvol.shape[1]
    rv = _rolling_sum(rep.inst_dvol, window)
    rn = _rolling_sum(rep.inst_pll, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        net_bps = np.where(rv > 0, 1e4 * rn / rv, np.nan)

    fig, axes, nrows = _grid_axes(n_inst, ncols, panel_h=1.6)
    days = rep.days
    for i in range(n_inst):
        ax = axes[i]
        y = net_bps[:, i]
        rate = 1e4 * rep.comm_rate[i]
        ax.plot(days, y, color="black", lw=0.9)
        ax.fill_between(days, 0, y, where=np.nan_to_num(y) >= 0, color=POS_C,
                        alpha=0.25, interpolate=True)
        ax.fill_between(days, 0, y, where=np.nan_to_num(y) < 0, color=NEG_C,
                        alpha=0.25, interpolate=True)
        ax.axhline(0, color="black", lw=0.6)
        ax.axhline(rate, color="#DD8452", ls="--", lw=0.9)
        if oos_start is not None:
            ax.axvline(oos_start, color="#ff7f0e", ls="--", lw=0.9, alpha=0.9)

        finite = y[np.isfinite(y)]
        if finite.size:
            lim = max(np.percentile(np.abs(finite), 98), rate * 2, 1e-3)
            ax.set_ylim(-lim, lim)
        name = "ALGO" if i == 0 else f"#{i}"
        overall = (1e4 * rep.inst_pll[:, i].sum() / rep.inst_dvol[:, i].sum()
                   if rep.inst_dvol[:, i].sum() > 0 else np.nan)
        ax.set_title(f"{name}   net {overall:+.2f} bp" if np.isfinite(overall)
                     else f"{name}   (no volume)",
                     fontsize=7, pad=2,
                     color=("#1a6b1a" if np.nan_to_num(overall) >= 0 else "#9b1c1c"))
        ax.tick_params(labelsize=6, length=2, pad=1)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    for i in range(n_inst):
        if i % ncols == 0:
            axes[i].set_ylabel("bps", fontsize=7)
        if i >= n_inst - ncols:
            axes[i].set_xlabel("day", fontsize=7)

    fig.suptitle(f"Rolling {window}d net PnL per dollar traded (bps) vs "
                 f"commission hurdle (orange dashed)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_file, dpi=140)
    return fig


def parse_args():
    p = argparse.ArgumentParser(description="Algothon IS/OOS backtester")
    p.add_argument("--strategy", default=DEFAULT_STRATEGY,
                   help="module name in this folder with getMyPosition (default: larpSharpe)")
    p.add_argument("--prices", default=PRICES_FILE, help="path to prices.txt")
    p.add_argument("--is-start", type=int, default=50)
    p.add_argument("--is-end", type=int, default=160)
    p.add_argument("--oos-start", type=int, default=160)
    p.add_argument("--oos-end", type=int, default=1000)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--top-n", type=int, default=10,
                   help="instruments shown as lines in the per-instrument PnL chart")
    p.add_argument("--quiet-strategy-path", action="store_true")

    # --- small-multiple grid + cost analytics ---
    p.add_argument("--grid-cols", type=int, default=6,
                   help="columns in the per-instrument grids (51 inst / 6 cols = 9 rows)")
    p.add_argument("--grid-roll", type=int, default=21,
                   help="rolling-mean window drawn over the daily PnL bars (0 = off)")
    p.add_argument("--grid-share-y", action="store_true",
                   help="share the y-axis across grid panels (comparable magnitudes)")
    p.add_argument("--inst-ind", type=int, default=None,
                   help="instrument index for the single-instrument cost series "
                        "(default: highest traded volume)")
    p.add_argument("--fee-window", type=int, default=63,
                   help="rolling window (days) for PnL-per-dollar-traded series")
    p.add_argument("--stat-window", choices=("full", "is", "oos", "last250"),
                   default="oos", help="window used for the cost bar chart / table")
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
    rep = calc_pl(prc, get_position, start_day=is_start, end_day=oos_end)
    days, pll, dvols = rep.days, rep.pll, rep.dvols
    tot_dvol, value, ret = rep.tot_dvol, rep.value, rep.ret
    inst_pll, util = rep.inst_pll, rep.util

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
    rep_o = calc_pl(prc, get_position, start_day=oos_start, end_day=oos_end)
    tot_o, val_o, ret_o = rep_o.tot_dvol, rep_o.value, rep_o.ret
    s_official = summarise(rep_o.pll, rep_o.dvols)
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

    mean_non_algo_util = 100 * util[:, 1:].mean()
    peak_day_util = 100 * util[:, 1:].mean(axis=1).max()
    print(f"Limit usage: non-ALGO mean {mean_non_algo_util:.1f}% of $10k caps, "
          f"busiest day {peak_day_util:.1f}%; ALGO mean {100*util[:,0].mean():.1f}% of $100k.")

    if s_official["tot_dvol"] < MIN_VOLUME_FLAG:
        print(f"WARN: official-window dollar volume {s_official['tot_dvol']:.0f} < {MIN_VOLUME_FLAG}.")

    # --- cost economics over the requested stat window ---
    stat_ranges = {
        "full": (is_start, oos_end),
        "is": (is_start, is_end),
        "oos": (oos_start, oos_end),
        "last250": (max(oos_end - 249, oos_start), oos_end),
    }
    lo, hi = stat_ranges[args.stat_window]
    ct = cost_table(rep, slice_mask(days, lo, hi))
    print_cost_table(ct, f"{args.stat_window} {lo}-{hi}")

    inst_ind = args.inst_ind
    if inst_ind is None:
        inst_ind = int(np.argmax(ct["vol"]))
    if not (0 <= inst_ind < n_inst):
        raise SystemExit(f"--inst-ind {inst_ind} outside 0..{n_inst - 1}")

    out_png = os.path.join(_SCRIPT_DIR, "backtest_is_oos.png")
    out_inst = os.path.join(_SCRIPT_DIR, "backtest_inst_pnl.png")
    out_util = os.path.join(_SCRIPT_DIR, "backtest_limit_usage.png")
    out_grid = os.path.join(_SCRIPT_DIR, "backtest_inst_daily_pnl_grid.png")
    out_feebar = os.path.join(_SCRIPT_DIR, "backtest_fee_per_volume_bars.png")
    out_feeone = os.path.join(_SCRIPT_DIR, f"backtest_cost_series_inst{inst_ind}.png")
    out_feegrid = os.path.join(_SCRIPT_DIR, "backtest_cost_series_grid.png")
    if not args.no_plot:
        fig_g = plot_daily_pnl_grid(days, inst_pll, out_grid, ncols=args.grid_cols,
                                    oos_start=oos_start, roll=args.grid_roll,
                                    share_y=args.grid_share_y)
        fig_fb = plot_fee_bars(ct, out_feebar, label=f"{args.stat_window} {lo}-{hi}")
        fig_f1 = plot_cost_series_single(rep, inst_ind, out_feeone,
                                         window=args.fee_window, oos_start=oos_start)
        fig_fg = plot_cost_series_grid(rep, out_feegrid, ncols=args.grid_cols,
                                       window=args.fee_window, oos_start=oos_start)
        fig_i = plot_instrument_pnl(days, inst_pll, out_inst, top_n=args.top_n)
        fig_u = plot_limit_usage(days, util, out_util)
        fig = plot_windows(days, pll, is_start, is_end, oos_start, oos_end, out_png)
        new_files = (out_grid, out_feebar, out_feeone, out_feegrid)
        for f in new_files:
            print(f"Saved plot -> {f}")
        backend = plt.get_backend().lower()
        if any(backend.startswith(b) for b in ("tkagg", "qt5agg", "qt4agg", "wxagg", "macosx")):
            print("Displaying plots (close windows to finish)...")
            plt.show(block=True)
        else:
            if sys.platform == "win32":
                for f in (out_png, out_inst, out_util) + new_files:
                    if os.path.exists(f):
                        os.startfile(os.path.abspath(f))
      #  for f_ in (fig_g, fig_fb, fig_f1, fig_fg, fig_i, fig_u, fig):
           # plt.close(f_)


if __name__ == "__main__":
    main()