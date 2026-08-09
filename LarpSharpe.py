"""Algothon 2026 v9 — v8 engine + null-calibrated roster gate & tiering.

MOTIVATION (measured on this file):
  v8's per-segment pair gate (t3<-2.9, t1,t2<-2.2) passes 113 ordered pairs;
  the SAME gate run on factor-null simulations (real PC1-3 factor paths +
  fresh independent random-walk idiosyncratics, i.e. zero true cointegration)
  passes 38-119.  The raw pass count is inside the null distribution, and the
  recency-score tail separates from the null tail only for the top ~15-20
  pairs.  A chunk of the old tier-3 book was statistically indistinguishable
  from noise — which is exactly the population whose "cointegration" failed
  to persist into Q7-8 (v8's own diagnosis).

DESIGN — sizing, not exclusion:
  A day-750 ablation (both rosters built ONLY on days 1-750, evaluated on the
  truly-unseen 750-1000, identical engine, sleeves off) showed:
    hard FDR exclusion (15 pairs):  score 316.3, sharpe 6.27  (caps bind;
                                    capacity cannot be redeployed)
    v8 gate, tercile tiers (32):    score 340.9, sharpe 5.28
    v8 gate, NULL-REF TIERS + probation LEG3=14k (chosen in-sample):
                                    score 353.5, sharpe 5.57   <-- v9 design
  Under score = mu*SR^2/(SR^2+1), capacity dominates once SR>3, so marginal
  pairs are kept but sized down; the risk budget concentrates in pairs the
  null calibration cannot explain.

ROSTER (rebuilt on full 1000d, reproducible via --rebuild-roster):
  Baskets: LassoCV stable-sign discovery per 330d segment, restricted-OLS
    gate ADF<-3.4 every segment, instruments 0 (ALGO leg would be discarded
    by the engine's hedge overwrite) and 24 (v8 curation) excluded from
    supports.  Honest null = SAME pipeline on null draws -> 0-3 spurious
    baskets, best rs -4.12.  6 real baskets kept: 4 clear the null bar
    (tier 1), 2 on probation.  NOTE: the earlier greedy-ADF null (adversarial
    partner selection) is the WRONG bar for LASSO-discovered baskets — it
    flunks 6/8 good ones.  Match the null to the actual selection procedure.
  Pairs: v8 per-segment gate for capacity, basket-subsumes-pair dedup,
    per-instrument concentration cap 5 (= v8 max), greedy by recency score.
    Tier edges from calibration: FDR<=10%% s* = -3.749, mean-null-best -3.693.
  TIER_PARAMS: LEG 30k / 22k / 14k (tier 3 = probation; 14k chosen in-sample
    in the ablation and confirmed OOS there).

Engine, sleeves (22/6/23), HEDGE_K=1.0, GROSS_SCALE=2.0 unchanged from v8.

CALIBRATED EXPECTATION: all selection still uses this file; apply the usual
decay prior (structural ~10-20%%, tuned params 35-45%%).  The ablation gives
the design +3.7%% score / +0.29 sharpe on truly unseen data vs v8's tiering.

Submission: rename to strategy.py (eval.py imports getMyPosition).
Dev tooling under __main__; roster rebuild: python strategy_v9.py --rebuild-roster
"""

import numpy as np

# ------------------------------------------------------------ competition
ALGO_IDX = 0
DEFAULT_CAP = 10_000.0
ALGO_CAP = 100_000.0
ACTIVITY_FLOOR = 50_000.0

FIT_LOOKBACK = 300
REFIT_EVERY = 5
MIN_HISTORY = 100

# ------------------------------------------------------------ sizing
GROSS_SCALE = 2.0    # TUNED: score rises monotonically to ~2.0 then plateaus
WEIGHT_SCOPE = "global"

# ------------------------------------------------------------ tiers
TIER_PARAMS = {
    1: dict(Z0=1.25, ZSTOP=4.0, LEG=30_000.0,
            W_LO=0.75, W_HI=1.5, HL_LO=2.0, HL_HI=80.0),
    2: dict(Z0=1.25, ZSTOP=4.0, LEG=22_000.0,
            W_LO=0.75, W_HI=1.5, HL_LO=2.0, HL_HI=80.0),
    3: dict(Z0=1.25, ZSTOP=4.0, LEG=14_000.0,   # probation size (v9)
            W_LO=0.75, W_HI=1.5, HL_LO=2.0, HL_HI=80.0),
}

# ------------------------------------------------------------ roster
# (tier, target, (partners...)) — target is regressed on partners.
# Built by the full-1000d scan described in the header.  Recency-weighted
# ADF shown in the scan log; tier 1 rs<-4.0, tier 2 rs<-3.45, tier 3 rest.
SPREADS = [
    # --- baskets (LASSO stable-sign discovery, excl inst 0/24;
    #     tier 1 iff rs beats best LASSO-pipeline null basket -4.119) ---
    (1, 45, (13, 15, 17, 35, 44, 48)),   # rs -5.815
    (1, 1, (20, 28)),   # rs -5.194
    (1, 10, (14, 46)),   # rs -4.930
    (1, 46, (3, 8, 10)),   # rs -4.804
    (3, 41, (36, 48, 49)),   # rs -4.109  [probation]
    (3, 39, (12, 17, 40, 43, 47, 49)),   # rs -3.996  [probation]
    # --- pairs (v8 per-segment gate for capacity; tiers by null calib:
    #     tier 1 rs < -3.749 (FDR<=10%), tier 2 rs < -3.693 (mean null best),
    #     tier 3 probation, sized at LEG3) ---
    (1, 20, (1,)),   # rs -5.221
    (1, 40, (7,)),   # rs -4.700
    (1, 37, (25,)),   # rs -4.639
    (1, 8, (27,)),   # rs -4.599
    (1, 13, (45,)),   # rs -4.386
    (1, 25, (37,)),   # rs -4.248
    (1, 7, (40,)),   # rs -4.234
    (1, 35, (18,)),   # rs -3.892
    (1, 18, (35,)),   # rs -3.761
    (2, 36, (41,)),   # rs -3.749
    (3, 41, (3,)),   # rs -3.606
    (3, 16, (6,)),   # rs -3.586
    (3, 33, (42,)),   # rs -3.568
    (3, 15, (45,)),   # rs -3.566
    (3, 40, (27,)),   # rs -3.543
    (3, 33, (49,)),   # rs -3.530
    (3, 33, (20,)),   # rs -3.519
    (3, 19, (2,)),   # rs -3.512
    (3, 36, (3,)),   # rs -3.498
    (3, 33, (12,)),   # rs -3.491
    (3, 8, (31,)),   # rs -3.469
    (3, 43, (31,)),   # rs -3.416
    (3, 2, (19,)),   # rs -3.413
    (3, 40, (12,)),   # rs -3.398
    (3, 33, (17,)),   # rs -3.369
    (3, 19, (10,)),   # rs -3.308
    (3, 49, (28,)),   # rs -3.306
    (3, 37, (43,)),   # rs -3.295
    (3, 28, (50,)),   # rs -3.261
    (3, 19, (38,)),   # rs -3.244
    (3, 9, (1,)),   # rs -3.244
    (3, 15, (4,)),   # rs -3.244
]

# ------------------------------------------------------------ EMA sleeve
# inst -> (fast, slow, dollars).  Residual-aware: sleeves only use capacity
# the spreads left behind.  Re-validated on the 1000d file (see dev log).
TREND_EMAS = {
    22: (25, 60, 10_000.0),   # MDGI — kept: leave-one-out -1.4 Q7-8 / -2.8 off.
    6:  (40, 90, 10_000.0),   # OTCS — kept: -11.0 Q7-8 / -4.8 off. when removed
    23: (20, 40, 10_000.0),   # AGVF — kept: -5.1 Q7-8 / -10.3 off. when removed
    # DROPPED on 1000d leave-one-out at final stack:
     #  21: (20,90, 10_000.0), # removing it GAINED +17.6 Q7-8 — flipped negative in the
    #      newest regime (also -1,672 in Q7-8 attribution under v7)
      # 11: (40,90, 10_000.0), # ~neutral (+1.6 Q7-8 removed) — dead weight, cut
    # Idle-scan on 26/34 (zero spread load): nothing (<50% +ve quarters)
}

# ------------------------------------------------------------ ALGO book
HEDGE_K = 1.0          # TUNED (reversal of v7's K=0): with the basket-heavy
                       # book the hedge now GAINS score AND Sharpe in Q7-8
                       # (513.8/6.56 vs 495-ish at K=0); K=0.75 ties within
                       # noise.  ALGO is therefore actively utilised again.
HEDGE_EMA_SPAN = 1
HEDGE_DEADBAND = 0.0
ALGO_TREND = dict(on=False, fast=30, slow=60, dollars=20_000.0)

# ------------------------------------------------------------ state
_state = {"last_fit": -10**9, "fits": [], "net_ema": None,
          "hedge_dlr": 0.0, "detail": None}


def reset_state():
    _state["last_fit"] = -10**9
    _state["fits"] = []
    _state["net_ema"] = None
    _state["hedge_dlr"] = 0.0
    _state["detail"] = None


# ------------------------------------------------------------ spread engine
def _fit_spread(px_win, tgt, partners, tp):
    """OLS of target on partners + exact discrete OU on the residual.
    None => benched this refit.  Pair == 1-partner special case."""
    y = px_win[tgt]
    X = px_win[list(partners)].T
    A = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    s = y - A @ coef
    yv, xv = s[1:], s[:-1]
    xm, ym = xv.mean(), yv.mean()
    vx = ((xv - xm) ** 2).sum()
    if vx <= 1e-12:
        return None
    phi = ((xv - xm) * (yv - ym)).sum() / vx
    c = ym - phi * xm
    if not (0.0 < phi < 0.9999):
        return None
    kappa = -np.log(phi)
    hl = np.log(2) / kappa
    if not (tp["HL_LO"] <= hl <= tp["HL_HI"]):
        return None
    eps = yv - (c + phi * xv)
    sigma_eq = eps.std() / np.sqrt(1.0 - phi ** 2)
    if sigma_eq <= 1e-9:
        return None
    mu = c / (1.0 - phi)
    capture = kappa * sigma_eq / max(y[-1], 1e-9)
    return dict(tgt=tgt, partners=tuple(partners),
                alpha=float(coef[0]), betas=coef[1:].astype(float),
                mu=float(mu), sigma=float(sigma_eq), cap=float(capture),
                hl=float(hl), w=1.0)


def _set_weights(fits_by_tier):
    """sqrt-capture weights, normalised globally or within tier."""
    if WEIGHT_SCOPE == "global":
        groups = [[f for fits in fits_by_tier.values() for f in fits]]
    else:
        groups = list(fits_by_tier.values())
    for grp in groups:
        caps = np.array([f["cap"] for f in grp])
        pos = caps[caps > 0]
        med = np.median(pos) if len(pos) else 1.0
        for f in grp:
            tp = TIER_PARAMS[f["tier"]]
            f["w"] = float(np.clip(np.sqrt(f["cap"] / med),
                                   tp["W_LO"], tp["W_HI"]))


def _refit(px, nt):
    win = px[:, -FIT_LOOKBACK:] if px.shape[1] > FIT_LOOKBACK else px
    fits_by_tier = {t: [] for t in TIER_PARAMS}
    for tier, tgt, partners in SPREADS:
        tp = TIER_PARAMS[tier]
        f = _fit_spread(win, tgt, partners, tp)
        if f is not None:
            f["tier"] = tier
            fits_by_tier[tier].append(f)
    _set_weights(fits_by_tier)
    _state["fits"] = [f for fits in fits_by_tier.values() for f in fits]
    _state["last_fit"] = nt


def _spread_dollars(fits, cur, px):
    """Requested dollars per instrument from all live spreads.  Returns
    per-spread share detail (sh_tgt, shares per partner) for attribution."""
    D = np.zeros(len(cur))
    detail = []
    for f in fits:
        tp = TIER_PARAMS[f["tier"]]
        tgt, ps = f["tgt"], f["partners"]
        s = px[tgt, -1] - (f["alpha"]
                           + float(f["betas"] @ px[list(ps), -1]))
        z = (s - f["mu"]) / f["sigma"]
        if abs(z) >= tp["ZSTOP"]:
            continue
        sgn = -np.clip(z / tp["Z0"], -1.0, 1.0)
        if sgn == 0.0:
            continue
        dt = sgn * tp["LEG"] * f["w"] * GROSS_SCALE
        sh_t = dt / cur[tgt]
        D[tgt] += dt
        sh_ps = []
        for k, b in zip(ps, f["betas"]):
            sh_k = -b * sh_t
            D[k] += sh_k * cur[k]
            sh_ps.append(sh_k)
        detail.append((f["tier"], tgt, ps, sh_t, tuple(sh_ps)))
    return D, detail


# ------------------------------------------------------------ EMA engine
def _ema_last(p, n):
    a = 2.0 / (n + 1)
    e = p[0]
    for v in p[1:]:
        e += a * (v - e)
    return e


def _trend_requests(px, n_inst):
    D = np.zeros(n_inst)
    for k, (fast, slow, dlr) in TREND_EMAS.items():
        g = _ema_last(px[k], fast) - _ema_last(px[k], slow)
        if g != 0.0:
            D[k] = np.sign(g) * dlr
    return D


def _algo_trend_dollars(px):
    if not ALGO_TREND["on"]:
        return 0.0
    g = (_ema_last(px[ALGO_IDX], ALGO_TREND["fast"])
         - _ema_last(px[ALGO_IDX], ALGO_TREND["slow"]))
    return float(np.sign(g) * ALGO_TREND["dollars"]) if g != 0.0 else 0.0


# ------------------------------------------------------------ main entry
def getMyPosition(prcSoFar):
    px = np.asarray(prcSoFar, dtype=float)
    nInst, nt = px.shape
    cur = px[:, -1]
    dcap = np.full(nInst, DEFAULT_CAP)
    dcap[ALGO_IDX] = ALGO_CAP
    share_cap = (dcap / cur).astype(int)
    if nt < MIN_HISTORY:
        return np.zeros(nInst, dtype=int)

    if nt - _state["last_fit"] >= REFIT_EVERY or not _state["fits"]:
        _refit(px, nt)

    # 1) spreads get first claim on every instrument's cap
    D_sp, sp_detail = _spread_dollars(_state["fits"], cur, px)
    D_sp_c = D_sp.copy()
    D_sp_c[1:] = np.clip(D_sp_c[1:], -dcap[1:], dcap[1:])

    # 2) EMA sleeve trades only the capacity spreads left behind
    resid = dcap - np.abs(D_sp_c)
    T_req = _trend_requests(px, nInst)
    T = np.clip(T_req, -resid, resid)
    T[ALGO_IDX] = 0.0

    D = D_sp_c + T
    D[1:] = np.clip(D[1:], -dcap[1:], dcap[1:])
    pos = np.clip(np.round(D / cur).astype(int), -share_cap, share_cap)

    # 3) ALGO book: smoothed/deadbanded hedge of net exposure + alpha sleeve
    net = float((pos * cur)[1:].sum())
    if HEDGE_EMA_SPAN > 1:
        a = 2.0 / (HEDGE_EMA_SPAN + 1)
        _state["net_ema"] = (net if _state["net_ema"] is None
                             else _state["net_ema"] + a * (net - _state["net_ema"]))
        net_s = _state["net_ema"]
    else:
        net_s = net
    hedge_target = -HEDGE_K * net_s
    if abs(hedge_target - _state["hedge_dlr"]) >= HEDGE_DEADBAND:
        _state["hedge_dlr"] = hedge_target
    algo_trend = _algo_trend_dollars(px)
    algo_dlr = _state["hedge_dlr"] + algo_trend
    pos[ALGO_IDX] = int(np.clip(round(algo_dlr / cur[ALGO_IDX]),
                                -share_cap[ALGO_IDX], share_cap[ALGO_IDX]))

    g = float(np.abs(pos * cur).sum())
    if 0 < g < ACTIVITY_FLOOR:
        pos = np.clip(np.round(pos * (ACTIVITY_FLOOR * 1.1 / g)).astype(int),
                      -share_cap, share_cap)

    _state["detail"] = dict(
        spreads=sp_detail,
        sleeve={k: T[k] / cur[k] for k in TREND_EMAS if T[k] != 0.0},
        hedge_sh=_state["hedge_dlr"] / cur[ALGO_IDX],
        algo_trend_sh=algo_trend / cur[ALGO_IDX],
    )
    return pos


# ============================================================ dev tooling
#  Everything below replays eval.py economics and adds attribution.

def _load_prices():
    import os
    import pandas as pd
    _dir = os.path.dirname(os.path.abspath(__file__))
    for fn in (os.path.join(_dir, "prices.txt"), "prices.txt"):
        if os.path.exists(fn):
            return pd.read_csv(fn, sep=r"\s+", header=0).values.T
    raise FileNotFoundError("prices.txt not found")


def _score(mu, sd):
    if mu <= 0 or sd < 1e-10:
        return float(mu)
    S = np.sqrt(250.0) * mu / sd
    return float(mu * S ** 2 / (S ** 2 + 1.0))


def _backtest(prc, start, end, attribution=False):
    """Replay eval.py economics over [start, end].  Resets strategy state."""
    reset_state()
    nInst = prc.shape[0]
    commRate = np.full(nInst, 0.0001); commRate[ALGO_IDX] = 0.00002
    dlrPosLimit = np.full(nInst, DEFAULT_CAP); dlrPosLimit[ALGO_IDX] = ALGO_CAP

    cash = value = comm = 0.0
    curPos = np.zeros(nInst)
    totDVolume = 0.0
    pll, gross_expo, daily_dvol, n_active = [], [], [], []

    inst_pnl = np.zeros(nInst)
    spread_pnl = {}                    # (tier,tgt,partners) -> cumulative $
    sleeve_pnl = {k: 0.0 for k in TREND_EMAS}
    hedge_pnl = 0.0
    algo_trend_pnl = 0.0
    prev_detail = None
    prev_prices = None
    prev_comm_vec = np.zeros(nInst)
    util_non_algo, util_algo = [], []
    util_by_inst = np.zeros(nInst); util_max = np.zeros(nInst); util_n = 0

    loop_start = max(start - 1, 1)
    for t in range(loop_start, end + 1):
        prcSoFar = prc[:, :t]
        curPrices = prcSoFar[:, -1]

        if t < end:
            newPos = getMyPosition(prcSoFar)
            posLimits = (dlrPosLimit / curPrices).astype(int)
            newPos = np.clip(newPos, -posLimits, posLimits).astype(int)
            detail = _state["detail"]
        else:
            newPos = np.array(curPos)
            detail = None

        if attribution and prev_prices is not None and t >= start:
            dp = curPrices - prev_prices
            inst_pnl += curPos * dp - prev_comm_vec
            if prev_detail is not None:
                for tier, tgt, ps, sh_t, sh_ps in prev_detail["spreads"]:
                    key = (tier, tgt, ps)
                    v = sh_t * dp[tgt] + sum(
                        sh * dp[k] for k, sh in zip(ps, sh_ps))
                    spread_pnl[key] = spread_pnl.get(key, 0.0) + v
                for k, sh in prev_detail["sleeve"].items():
                    sleeve_pnl[k] += sh * dp[k]
                hedge_pnl += prev_detail["hedge_sh"] * dp[ALGO_IDX]
                algo_trend_pnl += prev_detail["algo_trend_sh"] * dp[ALGO_IDX]

        deltaPos = newPos - curPos
        cash -= curPrices.dot(deltaPos) + comm
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = float(np.sum(dvolumes))
        totDVolume += dvolume
        prev_comm_vec = dvolumes * commRate
        comm = float(np.sum(prev_comm_vec))

        curPos = np.array(newPos)
        prev_detail = detail
        prev_prices = curPrices
        posValue = float(curPos.dot(curPrices))
        todayPL = cash + posValue - value
        value = cash + posValue

        if t >= start:
            pll.append(todayPL)
            dollars = np.abs(curPos * curPrices)
            gross_expo.append(float(dollars.sum()))
            daily_dvol.append(dvolume)
            n_active.append(int(np.count_nonzero(curPos)))
            if attribution:
                u = dollars / dlrPosLimit
                util_non_algo.append(float(u[1:].mean()))
                util_algo.append(float(u[ALGO_IDX]))
                util_by_inst += u; util_max = np.maximum(util_max, u)
                util_n += 1

    pll = np.asarray(pll, dtype=float)
    mu = float(pll.mean()) if len(pll) else 0.0
    sd = float(pll.std()) if len(pll) else 0.0  # ddof=0, matches eval.py
    sharpe = (np.sqrt(250.0) * mu / sd) if sd > 0 else 0.0
    equity = np.cumsum(pll) if len(pll) else np.array([0.0])
    drawdown = np.maximum.accumulate(equity) - equity

    out = {
        "n": len(pll), "mu": mu, "sd": sd, "sharpe": float(sharpe),
        "score": _score(mu, sd),
        "win_rate": float(np.mean(pll > 0)) if len(pll) else 0.0,
        "best": float(pll.max()) if len(pll) else 0.0,
        "worst": float(pll.min()) if len(pll) else 0.0,
        "max_dd": float(drawdown.max()) if len(pll) else 0.0,
        "tot_dvol": totDVolume,
        "avg_gross": float(np.mean(gross_expo)) if gross_expo else 0.0,
        "avg_turn": float(np.mean(daily_dvol)) if daily_dvol else 0.0,
        "avg_active": float(np.mean(n_active)) if n_active else 0.0,
        "final_value": value,
    }
    if attribution:
        out.update(dict(
            inst_pnl=inst_pnl, spread_pnl=spread_pnl, sleeve_pnl=sleeve_pnl,
            hedge_pnl=hedge_pnl, algo_trend_pnl=algo_trend_pnl,
            util_non_algo=float(np.mean(util_non_algo)) if util_non_algo else 0.0,
            util_algo=float(np.mean(util_algo)) if util_algo else 0.0,
            util_by_inst=util_by_inst / max(util_n, 1), util_max=util_max,
        ))
    return out


def _print_detail(label, s):
    print("=" * 46)
    print(label)
    print("=" * 46)
    print(f"{'Score':<20} {s['score']:>14.2f}")
    print(f"{'ann Sharpe':<20} {s['sharpe']:>14.2f}")
    print(f"{'mean(PL)':<20} {s['mu']:>14.2f}")
    print(f"{'std(PL)':<20} {s['sd']:>14.2f}")
    print(f"{'win rate':<20} {s['win_rate']:>13.1%}")
    print(f"{'best day':<20} {s['best']:>14.2f}")
    print(f"{'worst day':<20} {s['worst']:>14.2f}")
    print(f"{'max drawdown':<20} {s['max_dd']:>14.2f}")
    print(f"{'total $ volume':<20} {s['tot_dvol']:>14.0f}")
    print(f"{'avg gross expo':<20} {s['avg_gross']:>14.0f}")
    print(f"{'avg daily turnover':<20} {s['avg_turn']:>14.0f}")
    print(f"{'days':<20} {s['n']:>14d}")
    print(f"{'active (avg insts)':<20} {s['avg_active']:>14.1f}")
    if "util_non_algo" in s:
        print(f"{'util non-ALGO mean':<20} {s['util_non_algo']:>13.1%}")
        print(f"{'util ALGO mean':<20} {s['util_algo']:>13.1%}")
    print("=" * 46)


def _print_attribution(s):
    print("\nPer-spread attribution (gross, pre-commission), worst first:")
    print(f"{'tier':>5} {'spread':>34} {'PnL $':>12}")
    rows = sorted(s["spread_pnl"].items(), key=lambda kv: kv[1])
    tier_tot = {}
    for (tier, tgt, ps), v in rows:
        tier_tot[tier] = tier_tot.get(tier, 0.0) + v
        name = f"{tgt}~{list(ps)}"
        flag = "  <-- candidate bench" if v < 0 else ""
        print(f"{tier:>5} {name:>34} {v:>12.0f}{flag}")
    print("\nTier totals:")
    for tier in sorted(tier_tot):
        n = sum(1 for (tt, _, _) in s["spread_pnl"] if tt == tier)
        print(f"  tier {tier}: {tier_tot[tier]:>12.0f}   ({n} spreads fit)")
    print("\nSleeves / ALGO book (gross, pre-commission):")
    for k, v in s["sleeve_pnl"].items():
        print(f"  EMA sleeve inst {k:>2}: {v:>12.0f}")
    print(f"  ALGO hedge       : {s['hedge_pnl']:>12.0f}")
    print(f"  ALGO trend sleeve: {s['algo_trend_pnl']:>12.0f}")
    print("\nPer-instrument net PnL (incl. commission), losers only:")
    for k in np.argsort(s["inst_pnl"]):
        if s["inst_pnl"][k] < 0:
            print(f"  inst {k:>2}: {s['inst_pnl'][k]:>12.0f}"
                  f"   (mean util {s['util_by_inst'][k]:>5.1%},"
                  f" max {s['util_max'][k]:>5.1%})")
    idle = [k for k in range(len(s["inst_pnl"]))
            if s["util_max"][k] == 0.0]
    print(f"\nIdle instruments (never traded): {idle}")


def _windows(nt, qlen=125):
    """Auto-scaling window set sized off nt."""
    w = [
        ("last 250", nt - 250, nt),
        (f"official eval 160-{nt}", 160, nt),
        ("first 250 (pseudo-OOS)", 1, 1 + 250),
    ]
    quarters = []
    s = 1
    idx = 1
    while s < nt:
        e = min(s + qlen, nt)
        quarters.append((f"Q{idx} ({s}-{e})", s, e))
        s = e
        idx += 1
    w += quarters
    if len(quarters) >= 2:
        (_, s2, _), (_, _, e2) = quarters[-2], quarters[-1]
        w.append((f"Q{idx - 2}-{idx - 1} ({s2}-{nt}, newest unseen)",
                  s2, nt))
    return [(n, s, min(e, nt)) for n, s, e in w if s < nt]


def _run_report(numTestDays=250):
    prc = _load_prices()
    nInst, nt = prc.shape
    print(f"Loaded {nInst} instruments x {nt} days\n")

    detail_start = nt - numTestDays
    s250 = _backtest(prc, detail_start, nt, attribution=True)
    _print_detail(
        f"v8 backtest — last {numTestDays} days "
        f"[{detail_start}..{nt}] (eval.py mechanics)", s250)
    _print_attribution(s250)

    print(f"\n{'window':<34}{'score':>9}{'sharpe':>9}{'mu':>9}"
          f"{'util':>7}{'days':>7}")
    print("-" * 75)
    for name, s, e in _windows(nt):
        r = s250 if (s == detail_start and e == nt) else \
            _backtest(prc, s, e, attribution=True)
        print(f"{name:<34}{r['score']:>9.1f}{r['sharpe']:>9.2f}"
              f"{r['mu']:>9.1f}{r['util_non_algo']:>6.0%}{r['n']:>7d}")


# ------------------------------------------------------------ sweeps
def _sweep(prc, label, setter, values, windows=None):
    nt = prc.shape[1]
    wins = windows or [("Q1-4 (10-510)", 10, 510),
                       ("Q5-6 (510-760)", 510, 760),
                       ("Q7-8 newest (760-1000)", 760, nt),
                       (f"official 160-{nt}", 160, nt)]
    print(f"\nSweep: {label}")
    hdr = f"{'value':<16}" + "".join(f"{w[0]:>26}" for w in wins)
    print(hdr); print("-" * len(hdr))
    for v in values:
        setter(v)
        cells = []
        for _, s, e in wins:
            r = _backtest(prc, s, e)
            cells.append(f"{r['score']:>17.1f}/{r['sharpe']:>4.2f}   ")
        print(f"{str(v):<16}" + "".join(cells))


def _sweep_gross(prc):
    old = globals()["GROSS_SCALE"]
    try:
        _sweep(prc, "GROSS_SCALE (score/sharpe)",
               lambda v: globals().__setitem__("GROSS_SCALE", v),
               [1.0, 1.15, 1.3, 1.5])
    finally:
        globals()["GROSS_SCALE"] = old


def _sweep_hedge(prc):
    old = HEDGE_K
    try:
        _sweep(prc, "HEDGE_K (score/sharpe)",
               lambda v: globals().__setitem__("HEDGE_K", v),
               [0.0, 0.5, 1.0])
    finally:
        globals()["HEDGE_K"] = old


def _sweep_tiers(prc):
    olds = {t: TIER_PARAMS[t]["LEG"] for t in TIER_PARAMS}
    def setit(v):
        for t in TIER_PARAMS:
            TIER_PARAMS[t]["LEG"] = v[t - 1]
    try:
        _sweep(prc, "tier LEG sizes (t1,t2,t3) (score/sharpe)", setit,
               [(30_000.0, 22_000.0, 18_000.0),
                (34_000.0, 22_000.0, 14_000.0),
                (38_000.0, 24_000.0, 12_000.0),
                (30_000.0, 26_000.0, 22_000.0),
                (26_000.0, 20_000.0, 16_000.0)])
    finally:
        for t in TIER_PARAMS:
            TIER_PARAMS[t]["LEG"] = olds[t]


def _sweep_algo_ema(prc):
    import itertools
    old = dict(ALGO_TREND)
    def setit(v):
        if v == "off":
            ALGO_TREND.update(on=False)
        else:
            ALGO_TREND.update(on=True, fast=v[0], slow=v[1], dollars=v[2])
    try:
        grid = ["off"] + [(f, s, d)
                          for f, s in itertools.product((10, 20, 30, 40),
                                                        (40, 60, 90))
                          if f < s for d in (20_000.0, 40_000.0)]
        _sweep(prc, "ALGO_TREND (fast, slow, $) (score/sharpe)",
               setit, grid)
    finally:
        ALGO_TREND.clear(); ALGO_TREND.update(old)


# ------------------------------------------------------------ scans
def _adf_t(s):
    """Minimal ADF t-stat (no lag terms): regress ds on s_{t-1}."""
    ds = np.diff(s)
    x = s[:-1]
    X = np.column_stack([np.ones_like(x), x])
    coef, res, *_ = np.linalg.lstsq(X, ds, rcond=None)
    resid = ds - X @ coef
    dof = max(len(ds) - 2, 1)
    s2 = float(resid @ resid) / dof
    xm = x - x.mean()
    se = np.sqrt(s2 / float(xm @ xm)) if float(xm @ xm) > 0 else np.inf
    return float(coef[1] / se) if np.isfinite(se) and se > 0 else 0.0


def _scan_idle(prc, insts=(11, 12, 23, 26, 34, 38)):
    """EMA (fast,slow) trend sweep with per-quarter consistency on
    instruments with no spread load (sleeve candidates)."""
    nInst, nt = prc.shape
    print("EMA trend sweep, per-quarter mean daily PnL of "
          "sign(EMAf-EMAs)*$10k:")
    qs = [(s, min(s + 125, nt)) for s in range(125, nt - 1, 125)]
    for i in insts:
        best = None
        for fast in (10, 20, 30, 40):
            for slow in (40, 60, 90):
                if fast >= slow:
                    continue
                qmeans = []
                for s, e in qs:
                    pnl = []
                    for t in range(s, e - 1):
                        g = (_ema_last(prc[i, :t + 1], fast)
                             - _ema_last(prc[i, :t + 1], slow))
                        if g != 0.0:
                            sh = np.sign(g) * 10_000.0 / prc[i, t]
                            pnl.append(sh * (prc[i, t + 1] - prc[i, t]))
                    qmeans.append(np.mean(pnl) if pnl else 0.0)
                qmeans = np.array(qmeans)
                stat = (qmeans.mean(), float((qmeans > 0).mean()))
                if best is None or stat > best[0]:
                    best = (stat, fast, slow, qmeans)
        (m, frac), fast, slow, qmeans = best
        qtxt = " ".join(f"{v:>7.1f}" for v in qmeans)
        print(f"  inst {i:>2}: best EMA({fast},{slow})  mean/day {m:>7.1f}"
              f"  +ve quarters {frac:>4.0%}  [{qtxt}]")




# ============================================================ v9 roster lab
#  Null-calibrated gate tooling (dev only; not touched by eval.py import).
#  python strategy_v9.py --rebuild-roster   -> prints a fresh SPREADS block.


SEG_START = 10          # v8 convention: segments = thirds of [SEG_START, end]
HL_LO, HL_HI = 2.0, 80.0
HR_LO, HR_HI = 0.2, 4.0
FIT_LOOKBACK = 300      # trailing window for tradability prefilters (matches engine)
RS_W = (0.2, 0.3, 0.5)  # recency weights on (t1, t2, t3)


def segments(build_end: int):
    e1 = SEG_START + (build_end - SEG_START) // 3
    e2 = SEG_START + 2 * (build_end - SEG_START) // 3
    return [(SEG_START, e1), (e1, e2), (e2, build_end)]


# ---------------------------------------------------------------- vector ADF
def _adf_t_rows(S: np.ndarray) -> np.ndarray:
    """Row-wise ADF t (no lag terms), identical math to _adf_t."""
    ds = S[:, 1:] - S[:, :-1]
    x = S[:, :-1]
    xm = x.mean(axis=1, keepdims=True)
    dm = ds.mean(axis=1, keepdims=True)
    xc = x - xm
    vx = (xc * xc).sum(axis=1)
    cov = (xc * (ds - dm)).sum(axis=1)
    ok = vx > 1e-12
    b = np.where(ok, cov / np.where(ok, vx, 1.0), 0.0)
    resid = ds - dm - b[:, None] * xc
    dof = max(S.shape[1] - 1 - 2, 1)
    s2 = (resid * resid).sum(axis=1) / dof
    se = np.sqrt(np.where(ok, s2 / np.where(ok, vx, 1.0), np.inf))
    t = np.where((se > 0) & np.isfinite(se), b / se, 0.0)
    return t


def _resid_rows(Y: np.ndarray, x: np.ndarray):
    """OLS with intercept of every row of Y on x; returns (residuals, betas)."""
    xm = x.mean()
    xc = x - xm
    vx = float(xc @ xc)
    ym = Y.mean(axis=1, keepdims=True)
    beta = (Y - ym) @ xc / vx
    resid = Y - ym - beta[:, None] * xc[None, :]
    return resid, beta


# ---------------------------------------------------------------- pair scan
def pair_scan(px: np.ndarray, build_end: int):
    """All ordered pairs (target i regressed on partner j).

    Returns dict of (n,n) arrays: t1,t2,t3, rs, hl, hr, pre (tradability mask).
    Diagonal and failed fits are NaN/False.
    """
    n = px.shape[0]
    segs = segments(build_end)
    T = {k: np.full((n, n), np.nan) for k in ("t1", "t2", "t3")}
    trail = px[:, max(build_end - FIT_LOOKBACK, 0):build_end]

    hl = np.full((n, n), np.nan)
    hr = np.full((n, n), np.nan)

    for j in range(n):
        idx = np.array([i for i in range(n) if i != j])
        for k, (a, b) in enumerate(segs):
            S, _ = _resid_rows(px[idx, a:b], px[j, a:b])
            T[f"t{k+1}"][idx, j] = _adf_t_rows(S)
        # tradability on trailing window
        S, beta = _resid_rows(trail[idx], trail[j])
        phi_num = ((S[:, :-1] - S[:, :-1].mean(1, keepdims=True))
                   * (S[:, 1:] - S[:, 1:].mean(1, keepdims=True))).sum(1)
        phi_den = ((S[:, :-1] - S[:, :-1].mean(1, keepdims=True)) ** 2).sum(1)
        phi = np.where(phi_den > 1e-12, phi_num / np.where(phi_den > 1e-12, phi_den, 1), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            hl_j = np.where((phi > 0) & (phi < 0.9999), np.log(2) / -np.log(phi), np.nan)
        hl[idx, j] = hl_j
        hr[idx, j] = np.abs(beta) * trail[j, -1] / trail[idx, -1]

    rs = RS_W[0] * T["t1"] + RS_W[1] * T["t2"] + RS_W[2] * T["t3"]
    pre = (np.isfinite(rs) & np.isfinite(hl) & (hl >= HL_LO) & (hl <= HL_HI)
           & (hr >= HR_LO) & (hr <= HR_HI))
    np.fill_diagonal(pre, False)
    return dict(**T, rs=rs, hl=hl, hr=hr, pre=pre)


# ---------------------------------------------------------------- factor null
def fit_null_model(px: np.ndarray, build_end: int, k: int = 3):
    """PCA factor model on the build window: real factor paths + idio vols."""
    r = np.diff(np.log(px[:, :build_end]), axis=1)
    sd = r.std(axis=1); sd[sd < 1e-12] = 1e-12
    rs_ = r / sd[:, None]
    C = np.corrcoef(rs_)
    ev, V = np.linalg.eigh(C)
    B = V[:, ::-1][:, :k]                       # loadings (standardized space)
    f = B.T @ rs_                               # real factor return paths
    eps = rs_ - B @ f
    return dict(B=B, f=f, idio_sd=eps.std(axis=1), sd=sd, p0=px[:, 0].copy())


def simulate_null(model: dict, rng: np.random.Generator) -> np.ndarray:
    """Null panel: real factor paths + FRESH independent idio random walks.
    No pair is cointegrated (any combo retains a difference of indep RWs)."""
    B, f, isd, sd, p0 = (model[k] for k in ("B", "f", "idio_sd", "sd", "p0"))
    n, T = B.shape[0], f.shape[1]
    eps = rng.standard_normal((n, T)) * isd[:, None]
    r = sd[:, None] * (B @ f + eps)
    return p0[:, None] * np.exp(np.concatenate(
        [np.zeros((n, 1)), np.cumsum(r, axis=1)], axis=1))


# ---------------------------------------------------------------- FDR gate
def fdr_threshold(px: np.ndarray, build_end: int, n_null: int = 12,
                  q: float = 0.10, seed: int = 0, verbose: bool = True):
    """Largest (most permissive) rs threshold with estimated FDR <= q.

    FDR_hat(s) = mean_nulls #{null rs < s, prefilters passed} / #{real rs < s}.
    """
    real = pair_scan(px, build_end)
    r_sc = np.sort(real["rs"][real["pre"]])
    model = fit_null_model(px, build_end)
    rng = np.random.default_rng(seed)
    null_sc = []
    for _ in range(n_null):
        sc = pair_scan(simulate_null(model, rng), build_end)
        null_sc.append(np.sort(sc["rs"][sc["pre"]]))
    fp = np.array([[np.searchsorted(ns, s) for ns in null_sc] for s in r_sc]).mean(1)
    kept = np.arange(1, len(r_sc) + 1)
    fdr = fp / kept
    ok = np.where(fdr <= q)[0]
    s_star = r_sc[ok.max()] if len(ok) else -np.inf
    if verbose:
        nb = [ns[0] if len(ns) else np.nan for ns in null_sc]
        print(f"  [fdr] build_end={build_end} n_null={n_null}: real pass-pool "
              f"{len(r_sc)}, null best scores mean {np.nanmean(nb):.2f} "
              f"(min {np.nanmin(nb):.2f})")
        print(f"  [fdr] s* = {s_star:.3f} keeps {len(ok)} ordered pairs at "
              f"FDR<={q:.0%} (est. FP {fp[ok.max()] if len(ok) else 0:.1f})")
    return s_star, real, null_sc


# ---------------------------------------------------------------- builder
def build_roster(px: np.ndarray, build_end: int, gate: str,
                 max_pairs: int = 32, conc_cap: int = 6,
                 fdr_q: float = 0.10, n_null: int = 12, seed: int = 0,
                 tier_edges=None):
    """Greedy roster from ordered pairs. `gate`:
       'v8'   : t3<-2.9 and t1,t2<-2.2 (v8's per-segment thresholds)
       'null' : rs < s* from empirical-FDR calibration
    Both arms share prefilters, greedy order (rs asc), concentration cap,
    max size, and tiering rule, so they differ ONLY in the statistical gate.
    """
    if gate == "null":
        s_star, sc, null_sc = fdr_threshold(px, build_end, n_null, fdr_q, seed)
        mask = sc["pre"] & (sc["rs"] < s_star)
        nb = np.array([ns[0] if len(ns) else np.nan for ns in null_sc])
        n5 = np.array([ns[4] if len(ns) > 4 else np.nan for ns in null_sc])
        default_edges = (np.nanmean(nb), np.nanmean(n5))   # tier1 / tier2 cuts
    else:
        sc = pair_scan(px, build_end)
        mask = (sc["pre"] & (sc["t3"] < -2.9)
                & (sc["t1"] < -2.2) & (sc["t2"] < -2.2))
        default_edges = None

    cand = [(sc["rs"][i, j], int(i), int(j))
            for i, j in zip(*np.where(mask))]
    cand.sort()

    picked, appear = [], {}
    for rs_, i, j in cand:
        if len(picked) >= max_pairs:
            break
        if appear.get(i, 0) >= conc_cap or appear.get(j, 0) >= conc_cap:
            continue
        if any(t == i and p == j for _, t, p, _ in picked):
            continue
        picked.append((rs_, i, j, None))
        appear[i] = appear.get(i, 0) + 1
        appear[j] = appear.get(j, 0) + 1

    # tiering: shared rule = terciles by rs unless null-referenced edges given
    edges = tier_edges if tier_edges is not None else default_edges
    roster = []
    if edges is not None:
        e1, e2 = edges
        for rs_, i, j, _ in picked:
            tier = 1 if rs_ < e1 else (2 if rs_ < e2 else 3)
            roster.append((tier, i, (j,), rs_))
    else:
        rss = [p[0] for p in picked]
        t1e, t2e = (np.percentile(rss, [33.3, 66.7]) if rss else (0, 0))
        for rs_, i, j, _ in picked:
            tier = 1 if rs_ <= t1e else (2 if rs_ <= t2e else 3)
            roster.append((tier, i, (j,), rs_))
    return roster, sc


def roster_to_spreads(roster):
    return [(t, tgt, tuple(p)) for t, tgt, p, _ in roster]


def _lasso_discover(pxp, build_end, excl=(0, 24)):
    """v8-style basket discovery: LassoCV stable-sign support across the three
    segments, restricted-OLS gate ADF<-3.4 in every segment."""
    from sklearn.linear_model import LassoCV
    import warnings; warnings.filterwarnings("ignore")
    n = pxp.shape[0]
    segs = segments(build_end)
    def _adf3(tgt, partners):
        ts = []
        for a, b in segs:
            y = pxp[tgt, a:b]
            A = np.column_stack([np.ones(b - a)] + [pxp[p, a:b] for p in partners])
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            ts.append(_adf_t(y - A @ c))
        return ts
    found = []
    for tgt in range(n):
        if tgt in excl:
            continue
        cols = [i for i in range(n) if i != tgt and i not in excl]
        signs = []
        for a, b in segs:
            X = pxp[cols, a:b].T
            mu, sd = X.mean(0), X.std(0); sd[sd < 1e-12] = 1
            las = LassoCV(cv=3, n_alphas=40, max_iter=3000).fit((X - mu) / sd,
                                                                pxp[tgt, a:b])
            s = np.zeros(n); s[cols] = np.sign(las.coef_)
            signs.append(s)
        st = np.where((signs[0] != 0) & (signs[0] == signs[1])
                      & (signs[1] == signs[2]))[0]
        if not (2 <= len(st) <= 7):
            continue
        ts = _adf3(tgt, st)
        if all(t < -3.4 for t in ts):
            rs = RS_W[0]*ts[0] + RS_W[1]*ts[1] + RS_W[2]*ts[2]
            found.append((tgt, tuple(map(int, st)), rs))
    return found


def _rebuild_roster(build_end=None, n_null_pairs=12, n_null_baskets=4,
                    fdr_q=0.10, conc_cap=5, max_pairs=32, seed=0):
    """Full v9 roster rebuild.  Prints a paste-ready SPREADS block."""
    prc = _load_prices()
    if build_end is None:
        build_end = prc.shape[1]
    model = fit_null_model(prc, build_end)
    rng = np.random.default_rng(seed + 100)

    print("[1/3] basket discovery + honest LASSO-pipeline null ...")
    real_b = _lasso_discover(prc, build_end)
    null_rs = []
    for d in range(n_null_baskets):
        nb = _lasso_discover(simulate_null(model, rng), build_end)
        null_rs += [r for _, _, r in nb]
        print(f"    null draw {d}: {len(nb)} baskets pass")
    b_thr = min(null_rs) if null_rs else -4.0
    print(f"    basket tier-1 bar (best null rs): {b_thr:.3f}")

    print("[2/3] pair scan + FDR calibration ...")
    s_star, sc, null_sc = fdr_threshold(prc, build_end, n_null_pairs, fdr_q, seed)
    nb_p = float(np.mean([ns[0] for ns in null_sc]))
    t1e, t2e = min(s_star, nb_p), max(s_star, nb_p)

    print("[3/3] assembling ...")
    appear = {}
    baskets = []
    for tgt, ps, r in sorted(real_b, key=lambda z: z[2]):
        tier = 1 if r < b_thr else 3
        baskets.append((tier, tgt, ps, r))
        for k in (tgt,) + ps:
            appear[k] = appear.get(k, 0) + 1
    bsup = {tgt: set(ps) for _, tgt, ps, _ in baskets}
    mask = (sc["pre"] & (sc["t3"] < -2.9) & (sc["t1"] < -2.2) & (sc["t2"] < -2.2))
    for i in (0, 24):
        mask[i, :] = False; mask[:, i] = False
    pairs = []
    for r, i, j in sorted((sc["rs"][i, j], int(i), int(j))
                          for i, j in zip(*np.where(mask))):
        if len(pairs) >= max_pairs:
            break
        if j in bsup.get(i, ()):
            continue
        if appear.get(i, 0) >= conc_cap or appear.get(j, 0) >= conc_cap:
            continue
        tier = 1 if r < t1e else (2 if r < t2e else 3)
        pairs.append((tier, i, (j,), r))
        appear[i] = appear.get(i, 0) + 1
        appear[j] = appear.get(j, 0) + 1

    print("\nSPREADS = [")
    for t, tgt, ps, r in baskets:
        tag = "" if t == 1 else "  [probation]"
        print(f"    ({t}, {tgt}, {ps}),   # rs {r:.3f}{tag}")
    for t, tgt, ps, r in pairs:
        print(f"    ({t}, {tgt}, ({ps[0]},)),   # rs {r:.3f}")
    print("]")
    print(f"# pair edges: tier1 < {t1e:.3f}, tier2 < {t2e:.3f}; basket bar {b_thr:.3f}")
    return baskets, pairs


# ------------------------------------------------------------ CLI
if __name__ == "__main__":
    import sys
    args = set(sys.argv[1:])
    prc = _load_prices()
    if "--rebuild-roster" in args:
        _rebuild_roster()
    elif "--scan" in args:
        _scan_idle(prc)
    elif "--sweep-gross" in args:
        _sweep_gross(prc)
    elif "--sweep-hedge" in args:
        _sweep_hedge(prc)
    elif "--sweep-tiers" in args:
        _sweep_tiers(prc)
    elif "--sweep-algo-ema" in args:
        _sweep_algo_ema(prc)
    else:
        _run_report()