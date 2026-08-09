#!/usr/bin/env python
"""Algothon 2026 evaluation script.

Participants: write getMyPosition(prcSoFar) in teamName.py and update the imports below
"""

import os
import sys
import importlib

# Prefer an interactive backend so plt.show() actually pops up a window.
# Agg (common in IDE/CI) only saves files and cannot display.
if os.environ.get("MPLBACKEND", "").lower() in ("", "agg"):
    import matplotlib
    for backend in ("TkAgg", "Qt5Agg", "WXAgg"):
        try:
            matplotlib.use(backend, force=True)
            break
        except ImportError:
            continue

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sps

# Local default points at the strategy requested in this analysis.  Override
# without editing this file, e.g. ALGOTHON_STRATEGY=strategy python eval.py.
strategyModule = os.environ.get("ALGOTHON_STRATEGY", "strategy v6")
getPosition = importlib.import_module(strategyModule).getMyPosition

# anchor file paths to this script's folder so it runs from any directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

nInst = 0
nt = 0

pricesFile = os.path.join(_SCRIPT_DIR, "prices.txt")
testStartDay = 160   # first day included in score
testEndDay = 1000    # last released day (mark-only; no new trades)
plPlotFile = os.path.join(_SCRIPT_DIR, "eval_pl.png")
plDistPlotFile = os.path.join(_SCRIPT_DIR, "eval_pl_dist.png")
plBlockSize = 50

# parameter for scoring function
scoreDefaultParam = 1.0

# commission rates (0.0001 = 1bp)
# SPECIAL rate for instrument 0
defaultCommRate = 0.0001
inst0CommRate = 0.00002

# position limits (dollars)
# SPECIAL position limit for instrument 0
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000

def loadPrices(fn):
    """
    Load prices from csv file (one instrument per column) and transpose into one instrument per row
    """
    global nt, nInst
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    nt, nInst = df.shape
    return (df.values).T


def chargeFees(dvolumes, commRate):
    """
    Total commission for one day's trades.
    """
    return np.sum(dvolumes * commRate)


def score(mu, sigma, param=scoreDefaultParam):
    """
    Final score from the daily-PnL mean & std, plus a scoring parameter.
    """
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac

prcAll = loadPrices(pricesFile)
print(f"Loaded {nInst} instruments for {nt} days")

# initialise the per-instrument commissions and position limits
commRate = np.full(nInst, defaultCommRate)
commRate[0] = inst0CommRate
dlrPosLimit = np.full(nInst, defaultDlrPosLimit)
dlrPosLimit[0] = inst0DlrPosLimit

def calcPL(prcHist, testStartDay, testEndDay):
    """
    Function to loop over days and calculate/store PnLs
    """
    
    # initial values
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    value = 0
    comm = 0
    
    todayPLL = []
    days = []
    # day before testStartDay: first call to getPosition()
    startDay = testStartDay - 1
    
    for t in range(startDay, testEndDay + 1):
        # price history up to and including t, e.g. if t=500, gets first 500 days
        prcHistSoFar = prcHist[:, :t]
        curPrices = prcHistSoFar[:, -1]

        # trading loop, do not do it on the very last day of the test
        if t < testEndDay:
            # get new positions
            newPosOrig = getPosition(prcHistSoFar)

            # clip to position limits, and enforce integer shares
            posLimits = (dlrPosLimit / curPrices).astype(int)
            newPos = np.clip(newPosOrig, -posLimits, posLimits).astype(int)
        else:
            # the final day is only used as 'mark' of final PnL
            newPos = np.array(curPos)

        # change in positions
        deltaPos = newPos - curPos
        
        cash -= curPrices.dot(deltaPos) + comm

        # calculate commissions
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume
        comm = chargeFees(dvolumes, commRate)
            
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        # PnL is the daily change in portfolio value (cash plus positions)
        todayPL = cash + posValue - value
        
        value = cash + posValue

        # calculate return (portfolio value over total dollar volume)
        ret = 0.0
        if totDVolume > 0:
            ret = value / totDVolume

        # only score for test days
        if t >= testStartDay:
            print(
                f"Day {t} value: {value:.2f} todayPL: ${todayPL:.2f} $-traded: {totDVolume:.0f} return: {ret:.5f}"
            )
            todayPLL.append(todayPL)
            days.append(t)
            
    pll = np.array(todayPLL)
    days = np.array(days)
    plmu, plstd = (np.mean(pll), np.std(pll))

    # calculate annualised Sharpe
    annSharpe = 0.0
    if plstd > 0:
        annSharpe = np.sqrt(250) * plmu / plstd
        
    return (plmu, ret, plstd, annSharpe, totDVolume, days, pll)

def plotPL(days, pll, testStartDay, testEndDay, outFile, blockSize=plBlockSize):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(days, pll, width=0.8, color=np.where(pll >= 0, "#2ca02c", "#d62728"), alpha=0.85)
    plByBlock = (
        pd.DataFrame({"day": days, "pl": pll})
        .assign(block=lambda d: (d["day"] - testStartDay) // blockSize)
    )
    blockScore = (
        plByBlock.groupby("block", group_keys=False)["pl"]
        .transform(lambda g: score(g.mean(), g.std()))
    )
    blockMean = (
        plByBlock.groupby("block", group_keys=False)["pl"]
        .transform("mean")
    )
    ax.plot(days, blockScore, color="#1f77b4", linewidth=2,
            label=f"{blockSize}-day block score")
    ax.plot(days, blockMean, color="#ff7f0e", linewidth=2,
            label=f"{blockSize}-day block mean")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Daily PnL ($)")
    ax.set_title(f"Daily PnL (days {testStartDay}-{testEndDay})")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outFile, dpi=150)
    print(f"Saved PnL plot to {outFile}")
    return fig


def plDistributionStats(pll):
    """Moments + normality tests for daily PnL."""
    n = len(pll)
    mu = float(np.mean(pll))
    sigma = float(np.std(pll, ddof=1)) if n > 1 else 0.0
    skew = float(sps.skew(pll, bias=False)) if n > 2 else float("nan")
    kurt = float(sps.kurtosis(pll, fisher=True, bias=False)) if n > 3 else float("nan")
    # excess kurtosis: 0 = normal; >0 = fat tails
    jb_stat, jb_p = sps.jarque_bera(pll) if n >= 2 else (float("nan"), float("nan"))
    # Shapiro-Wilk is reliable for n <= ~5000; our eval window is ~340 days
    if 3 <= n <= 5000:
        sw_stat, sw_p = sps.shapiro(pll)
    else:
        sw_stat, sw_p = float("nan"), float("nan")
    # SciPy >=1.17: prefer method= so we get a p-value without FutureWarning
    try:
        ad = sps.anderson(pll, dist="norm", method="interpolate")
        ad_stat = float(ad.statistic)
        ad_p = float(ad.pvalue) if hasattr(ad, "pvalue") else float("nan")
        ad_crit_5 = float("nan")
    except TypeError:
        ad = sps.anderson(pll, dist="norm")
        ad_stat = float(ad.statistic)
        ad_p = float("nan")
        # critical values at 5% (index 2 in scipy's default levels)
        ad_crit_5 = float(ad.critical_values[2]) if len(ad.critical_values) > 2 else float("nan")
    return {
        "n": n,
        "mean": mu,
        "std": sigma,
        "skew": skew,
        "excess_kurtosis": kurt,
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_p": float(jb_p),
        "shapiro_stat": float(sw_stat),
        "shapiro_p": float(sw_p),
        "anderson_stat": ad_stat,
        "anderson_p": ad_p,
        "anderson_crit_5pct": ad_crit_5,
    }


def plotPLDistribution(pll, testStartDay, testEndDay, outFile):
    """Histogram + KDE + normal overlay, with distribution / normality stats."""
    stats = plDistributionStats(pll)
    mu, sigma = stats["mean"], stats["std"]
    n = stats["n"]

    fig, ax = plt.subplots(figsize=(10, 6))
    n_bins = max(10, min(40, int(np.sqrt(n))))
    ax.hist(pll, bins=n_bins, density=True, color="#7f7f7f", alpha=0.55,
            edgecolor="white", label="Daily PnL")

    xs = np.linspace(pll.min(), pll.max(), 300)
    if sigma > 1e-12:
        ax.plot(xs, sps.norm.pdf(xs, mu, sigma), color="#1f77b4", linewidth=2.2,
                label="Normal(μ, σ)")
    try:
        kde = sps.gaussian_kde(pll)
        ax.plot(xs, kde(xs), color="#d62728", linewidth=2.0, label="KDE")
    except Exception:
        pass

    ax.axvline(mu, color="#2ca02c", linewidth=1.5, linestyle="--", label=f"mean ${mu:.0f}")
    ax.axvline(0, color="black", linewidth=0.8)

    tests = [stats["shapiro_p"], stats["jarque_bera_p"]]
    if not np.isnan(stats["anderson_p"]):
        tests.append(stats["anderson_p"])
    normal_ok = all((not np.isnan(p) and p >= 0.05) for p in tests)
    verdict = "consistent with normal (p>=0.05)" if normal_ok else "NOT normal (reject at 5%)"
    if not np.isnan(stats["anderson_p"]):
        ad_line = f"Anderson p={stats['anderson_p']:.4g}"
    else:
        ad_line = (
            f"Anderson A2={stats['anderson_stat']:.3f} "
            f"(5% crit {stats['anderson_crit_5pct']:.3f})"
        )
    box = (
        f"n={stats['n']}\n"
        f"mean ${stats['mean']:.1f}\n"
        f"std  ${stats['std']:.1f}\n"
        f"skew {stats['skew']:+.3f}\n"
        f"exKurt {stats['excess_kurtosis']:+.3f}\n"
        f"Shapiro p={stats['shapiro_p']:.4g}\n"
        f"Jarque-Bera p={stats['jarque_bera_p']:.4g}\n"
        f"{ad_line}\n"
        f"-> {verdict}"
    )
    ax.text(
        0.98, 0.98, box, transform=ax.transAxes, va="top", ha="right",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )
    ax.set_xlabel("Daily PnL ($)")
    ax.set_ylabel("Density")
    ax.set_title(f"Daily PnL distribution (days {testStartDay}-{testEndDay})")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outFile, dpi=150)
    print(f"Saved PnL distribution plot to {outFile}")
    return stats, fig


meanpl, ret, plstd, sharpe, dvol, days, pll = calcPL(prcAll, testStartDay, testEndDay)
fig_pl = plotPL(days, pll, testStartDay, testEndDay, plPlotFile)
distStats, fig_dist = plotPLDistribution(pll, testStartDay, testEndDay, plDistPlotFile)
scoreVal = score(meanpl, plstd, scoreDefaultParam)
print("=====")
print(f"mean(PL): {meanpl:.1f}")
print(f"return: {ret:.5f}")
print(f"StdDev(PL): {plstd:.2f}")
print(f"annSharpe(PL): {sharpe:.2f}")
print(f"totDvolume: {dvol:.0f}")
print(f"Score: {scoreVal:.2f}")
print("----- PnL distribution -----")
print(f"n: {distStats['n']}")
print(f"skew: {distStats['skew']:+.3f}")
print(f"excess kurtosis: {distStats['excess_kurtosis']:+.3f}  (0 ~ normal; >0 fat tails)")
print(f"Shapiro-Wilk: W={distStats['shapiro_stat']:.4f}  p={distStats['shapiro_p']:.4g}")
print(f"Jarque-Bera:  JB={distStats['jarque_bera_stat']:.3f}  p={distStats['jarque_bera_p']:.4g}")
if not np.isnan(distStats["anderson_p"]):
    print(
        f"Anderson-Darling: A2={distStats['anderson_stat']:.3f}  "
        f"p={distStats['anderson_p']:.4g}"
    )
else:
    print(
        f"Anderson-Darling: A2={distStats['anderson_stat']:.3f}  "
        f"5% critical={distStats['anderson_crit_5pct']:.3f}"
    )
_ad_ok = (
    (not np.isnan(distStats["anderson_p"]) and distStats["anderson_p"] >= 0.05)
    or (
        np.isnan(distStats["anderson_p"])
        and distStats["anderson_stat"] < distStats["anderson_crit_5pct"]
    )
)
if (
    distStats["shapiro_p"] >= 0.05
    and distStats["jarque_bera_p"] >= 0.05
    and _ad_ok
):
    print("Normality: FAIL TO REJECT at 5% (looks roughly Gaussian)")
else:
    print("Normality: REJECT at 5% (skew / fat tails / outliers matter)")

# Show figures in a window; if the backend is non-interactive (Agg),
# open the saved PNGs with the OS default viewer instead.

_backend = plt.get_backend().lower()
_interactive_backends = (
    "tkagg", "qt5agg", "qt4agg", "wxagg", "macosx", "gtk3agg", "gtk4agg",
)
if any(_backend.startswith(name) for name in _interactive_backends):
    print("Displaying PnL plots (close the window to finish)...")
    plt.show(block=True)
else:
    no_open = os.environ.get("ALGOTHON_NO_OPEN", "").lower() in (
        "1", "true", "yes",
    )
    print(f"Matplotlib backend is {_backend!r} (non-interactive).")
    if not no_open:
        for path in (plPlotFile, plDistPlotFile):
            if os.path.exists(path):
                if sys.platform == "win32":
                    os.startfile(os.path.abspath(path))
                elif sys.platform == "darwin":
                    os.system(f'open "{path}"')
                else:
                    os.system(f'xdg-open "{path}"')
