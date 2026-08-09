"""Visual sanity check for prices.txt.

Reads the whitespace-delimited price matrix (500 days x 51 instruments,
header row of tickers) and produces three figures:
  1. A grid of individual price plots, one per instrument (labelled).
  2. All 51 instruments overlaid on one axes (absolute prices).
  3. All 51 instruments overlaid, each normalised to its day-0 price
     (relative increase).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRICES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "algothon26-starter-code-main", "prices.txt")


def main():
    df = pd.read_csv(PRICES_FILE, sep=r"\s+", header=0)
    names = list(df.columns)
    prices = df.values  # shape: (nDays, nInstruments)
    n_days, n_inst = prices.shape
    days = np.arange(n_days)

    # Figure 1: individual plots, one panel per instrument
    n_cols = 6
    n_rows = int(np.ceil(n_inst / n_cols))
    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(14, 16), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, ax in enumerate(axes.flat):
        if i < n_inst:
            ax.plot(days, prices[:, i], linewidth=0.7,
                    color=colors[i % len(colors)])
            ax.set_title(names[i], fontsize=7)
            ax.tick_params(labelsize=5)
        else:
            ax.axis("off")
    fig1.suptitle(f"Individual prices: {n_inst} instruments, {n_days} days",
                  fontsize=11)
    fig1.tight_layout(rect=[0, 0, 1, 0.98])
    fig1.savefig("visualcheck_individual.png", dpi=150)

    # Figure 2: all instruments stacked on one axes (absolute prices)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(days, prices, linewidth=0.8)
    ax2.set_title(f"All {n_inst} instruments over time (absolute prices)")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Price")
    fig2.tight_layout()
    fig2.savefig("visualcheck_stacked.png", dpi=150)

    # Figure 3: relative increase (each instrument normalised to day 0)
    relative = prices / prices[0, :]
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(days, relative, linewidth=0.8)
    ax3.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax3.set_title(f"All {n_inst} instruments over time "
                  "(normalised to day-0 price)")
    ax3.set_xlabel("Day")
    ax3.set_ylabel("Relative price (day 0 = 1.0)")
    fig3.tight_layout()
    fig3.savefig("visualcheck_relative.png", dpi=150)

    plt.show()


if __name__ == "__main__":
    main()
