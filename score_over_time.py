#!/usr/bin/env python
"""Score-over-time chart for a strategy: 8-quarter discrete/line score plus a
daily rolling continuous score, on one axis (both series are score(mu, sigma)
in dollars, eval.py's scoring function, just at two temporal resolutions).

Run: python score_over_time.py
Override strategy/window with env vars, e.g.:
  ALGOTHON_STRATEGY="strategy v6" ROLLING_WINDOW=50 python score_over_time.py
"""

import os
import importlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

strategyModule = os.environ.get("ALGOTHON_STRATEGY", "strategy v8")
getPosition = importlib.import_module(strategyModule).getMyPosition

pricesFile = os.path.join(_SCRIPT_DIR, "prices.txt")
testStartDay = 2   # matches eval.py's official scoring window
testEndDay = 1002
rollingWindow = int(os.environ.get("ROLLING_WINDOW", 125))
nQuarters = 8
outFile = os.path.join(_SCRIPT_DIR, "score_over_time.png")

# commission rates (0.0001 = 1bp), SPECIAL rate for instrument 0 -- eval.py convention
defaultCommRate = 0.0001
inst0CommRate = 0.00002
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000


def loadPrices(fn):
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    nt, nInst = df.shape
    return (df.values).T, nInst


def score(mu, sigma, param=1.0):
    """eval.py's score function: Sharpe-shrunk mean daily PnL."""
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac


prcAll, nInst = loadPrices(pricesFile)

commRate = np.full(nInst, defaultCommRate)
commRate[0] = inst0CommRate
dlrPosLimit = np.full(nInst, defaultDlrPosLimit)
dlrPosLimit[0] = inst0DlrPosLimit


def calcPL(prcHist, testStartDay, testEndDay):
    cash = 0.0
    curPos = np.zeros(nInst)
    value = 0.0
    comm = 0.0
    todayPLL = []
    days = []
    startDay = testStartDay - 1

    for t in range(startDay, testEndDay + 1):
        prcHistSoFar = prcHist[:, :t]
        curPrices = prcHistSoFar[:, -1]

        if t < testEndDay:
            newPosOrig = getPosition(prcHistSoFar)
            posLimits = (dlrPosLimit / curPrices).astype(int)
            newPos = np.clip(newPosOrig, -posLimits, posLimits).astype(int)
        else:
            newPos = np.array(curPos)

        deltaPos = newPos - curPos
        cash -= curPrices.dot(deltaPos) + comm
        dvolumes = curPrices * np.abs(deltaPos)
        comm = np.sum(dvolumes * commRate)

        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue

        if t >= testStartDay:
            todayPLL.append(todayPL)
            days.append(t)

    return np.array(days), np.array(todayPLL)


print(f"Backtesting {strategyModule!r} on days {testStartDay}-{testEndDay}...")
days, pll = calcPL(prcAll, testStartDay, testEndDay)

# ---- 8-quarter discrete score (fixed 125-day windows: 1-125, 126-250, ..., 876-1000) ----
periodLen = (testEndDay - testStartDay + 1) // nQuarters
qBounds = [(testStartDay + i * periodLen, testStartDay + (i + 1) * periodLen - 1)
           for i in range(nQuarters)]
qBounds[-1] = (qBounds[-1][0], testEndDay)  # last window absorbs any remainder days
qIdx = [np.where((days >= lo) & (days <= hi))[0] for lo, hi in qBounds]
qDay = np.array([days[idx].mean() for idx in qIdx])
qScore = np.array([score(pll[idx].mean(), pll[idx].std()) for idx in qIdx])

# ---- daily rolling continuous score ----
rollDays, rollScore = [], []
for i in range(rollingWindow - 1, len(days)):
    window = pll[i - rollingWindow + 1: i + 1]
    rollDays.append(days[i])
    rollScore.append(score(window.mean(), window.std()))
rollDays = np.array(rollDays)
rollScore = np.array(rollScore)

officialScore = score(pll.mean(), pll.std())
print(f"Official score (days {testStartDay}-{testEndDay}): {officialScore:.1f}")
for i, idx in enumerate(qIdx):
    print(f"  Q{i + 1} (day {days[idx[0]]}-{days[idx[-1]]}): score {qScore[i]:.1f}")

# ---- plot (validated categorical slots 1 & 2: blue / orange) ----
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

fig, ax = plt.subplots(figsize=(12, 6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

ax.plot(rollDays, rollScore, color=BLUE, linewidth=2, zorder=2,
        label=f"Daily score ({rollingWindow}d trailing window)")
ax.plot(qDay, qScore, color=ORANGE, linewidth=2, zorder=3)
ax.scatter(qDay, qScore, color=ORANGE, s=70, zorder=4, edgecolor=SURFACE,
           linewidth=1.2, label="Quarterly score (8 quarters)")

ax.axhline(0, color=BASELINE, linewidth=1)
ax.set_xlabel("Day", color=SECONDARY_INK)
ax.set_ylabel("Score  (score(μ, σ) on daily PnL, $)", color=SECONDARY_INK)
ax.set_title(f"{strategyModule} — score over time (days {testStartDay}-{testEndDay})",
             color=INK, fontsize=13, fontweight="bold")

ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(BASELINE)
ax.tick_params(colors=MUTED)

ax.legend(loc="upper left", frameon=False, labelcolor=SECONDARY_INK)

fig.tight_layout()
fig.savefig(outFile, dpi=150)
print(f"Saved plot to {outFile}")