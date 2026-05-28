"""
Marcus Theory of Capital Rotation — Data Generation Module
Generates synthetic Cholesky-correlated GBM returns with GARCH volatility
and a sector momentum injection schedule designed to traverse all three
Marcus regions (normal, activationless, inverted).
"""

import numpy as np
from config import CONFIG, SECTOR_NAMES


def _rolling_var(x, window):
    """Population rolling variance via cumulative-sum trick. O(T)."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < window:
        return out
    cs  = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x ** 2)])
    s1 = cs[window:] - cs[:-window]
    s2 = cs2[window:] - cs2[:-window]
    out[window - 1:] = s2 / window - (s1 / window) ** 2
    return np.maximum(out, 0.0)


def _build_injection_schedule(T, N_s):
    """
    Returns (T, N_s) array of daily mean log-returns per sector.
    Designed to traverse: normal -> activationless -> inverted -> crisis
    -> recovery -> inverted-2 -> activationless.
    """
    sched = np.zeros((T, N_s))
    # Sector indices: 0=Tech, 1=Fin, 2=Health, 3=Energy, 4=ConsDisc, 5=Indust

    # -- Phase 1: NORMAL REGION (days 0-119) --
    sched[:120, 0] = 0.0012    # Tech  +0.12 %/day
    sched[:120, 3] = 0.0003    # Energy +0.03 %/day

    # -- Phase 2: APPROACHING ACTIVATIONLESS (days 120-199) --
    for t in range(120, 200):
        p = (t - 120) / 80.0
        sched[t, 0] = 0.0012 + p * (0.0030 - 0.0012)
        sched[t, 3] = 0.0003 + p * (-0.0005 - 0.0003)

    # -- Phase 3: INVERTED REGION EPISODE 1 (days 200-339) --
    sched[200:340, 0] = 0.0045   # Tech  +0.45 %/day
    sched[200:340, 3] = -0.0005  # Energy -0.05 %/day

    # -- Phase 4: CRISIS & REBALANCING (days 340-449) --
    sched[340:450, 0] = 0.0010
    sched[340:450, 3] = 0.0010

    # -- Phase 5: RECOVERY - oscillate (days 450-559) --
    for t in range(450, 560):
        cyc = np.sin(2 * np.pi * (t - 450) / 40.0)
        sched[t, 0] = 0.0015 + 0.0008 * cyc
        sched[t, 3] = 0.0005 - 0.0008 * cyc

    # -- Phase 6: INVERTED REGION EPISODE 2 (days 560-659) --
    # Energy/Industrials now expensive; Tech cheap
    sched[560:660, 3] = 0.0030
    sched[560:660, 5] = 0.0025
    sched[560:660, 0] = -0.0010

    # -- Phase 7: ACTIVATIONLESS PERIOD (days 660-755) --
    sched[660:756, 0] = 0.0018
    sched[660:756, 3] = 0.0008

    return sched


def _apply_injection(log_returns, injection, sector_indices, Ns, sps):
    """Add sector injection drifts to stock returns. Broadcasting-safe."""
    out = log_returns.copy()
    for s in range(Ns):
        out[:, sector_indices[s]] += injection[:, s:s+1]  # (T,1) broadcasts over (T,5)
    return out


def _compute_sector_returns(log_returns, sector_indices, Ns, sps):
    """Equal-weighted sector daily log-returns from stock returns."""
    T = log_returns.shape[0]
    sec_ret = np.zeros((T, Ns))
    for s in range(Ns):
        sec_ret[:, s] = np.mean(log_returns[:, sector_indices[s]], axis=1)
    return sec_ret


def _compute_momentum(sector_returns, ROLL):
    """
    Rolling W-day sector momentum via cumulative-sum trick.
    Returns shape (T - ROLL, Ns) — strictly T-W values for days W to T-1.
    """
    T, Ns = sector_returns.shape
    cum = np.vstack([np.zeros((1, Ns)), np.cumsum(sector_returns, axis=0)])  # (T+1, Ns)
    # FIX: cum[W:T] - cum[0:T-W] gives exactly T-W values (days W to T-1)
    # Previously cum[W:] - cum[:-W] gave T-W+1 values (included phantom day T)
    return cum[ROLL:T] - cum[0:T - ROLL]


def _compute_calibration_metrics(sector_returns, c_dg0, c_lam, ROLL, Ns):
    """Compute calibrated dG0 and lambda for ratio checking."""
    T = sector_returns.shape[0]
    mom = _compute_momentum(sector_returns, ROLL)  # (T-ROLL, Ns)

    sorted_mom = np.sort(mom, axis=1)
    m_bot2 = (sorted_mom[:, 0] + sorted_mom[:, 1]) / 2.0
    m_top2 = (sorted_mom[:, -1] + sorted_mom[:, -2]) / 2.0
    dG0_raw = -0.5 * (m_top2 - m_bot2)  # (T-ROLL,)

    sec_var = np.zeros((T, Ns))
    for s in range(Ns):
        sec_var[:, s] = _rolling_var(sector_returns[:, s], CONFIG["ROLL_LAMBDA"])

    # axis=1 averages ACROSS sectors, producing a time series (T-ROLL,)
    lam_raw = np.mean(sec_var[ROLL:], axis=1)

    dG0_cal = dG0_raw * c_dg0
    lam_cal = lam_raw * c_lam

    valid = np.isfinite(dG0_cal) & np.isfinite(lam_cal) & (lam_cal > 0)
    med_dG0 = np.median(np.abs(dG0_cal[valid]))
    med_lam = np.median(lam_cal[valid])

    n_inv = int(np.sum((dG0_cal < -lam_cal) & valid))
    ratio = med_dG0 / (med_lam + 1e-20)

    return n_inv, ratio, dG0_cal, lam_cal


def generate_returns():
    """
    Generate synthetic stock returns and derived sector data.

    Returns
    -------
    stock_returns : (T, N_STOCKS) daily log-returns
    sector_returns : (T, N_SECTORS) equal-weighted sector log-returns
    sector_indices : list of lists, sector_indices[s] = stock indices in sector s
    c_dg0_cal : calibrated driving-force scaling
    c_lam_cal : calibrated reorganisation-energy scaling
    """
    C = CONFIG
    rng = np.random.RandomState(C["SEED"])
    N  = C["N_STOCKS"]
    T  = C["T_TOTAL"]
    Ns = C["N_SECTORS"]
    sps = C["STOCKS_PER_SECTOR"]
    ROLL = C["ROLL_MOMENTUM"]

    # -- Build block correlation matrix --
    corr = np.full((N, N), C["RHO_ACROSS"])
    for s in range(Ns):
        i0, i1 = s * sps, (s + 1) * sps
        corr[i0:i1, i0:i1] = C["RHO_WITHIN"]
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr)

    # -- GARCH(1,1) variance paths (Trap-4 safe: independent shocks) --
    sigma2_daily = (C["SIGMA_BASE"] ** 2) / 252.0
    garch_shocks = rng.randn(T, N)
    sigma2 = np.full((T, N), sigma2_daily)
    for t in range(1, T):
        sigma2[t] = (C["GARCH_OMEGA"]
                     + C["GARCH_ALPHA"] * garch_shocks[t - 1] ** 2 * sigma2_daily
                     + C["GARCH_BETA"] * sigma2[t - 1])
        sigma2[t] = np.clip(sigma2[t], sigma2_daily * 0.1, sigma2_daily * 100.0)

    # -- Correlated standard normals --
    Z = rng.randn(T, N)
    Z_corr = Z @ L.T

    # -- Volatility multiplier for crisis phase --
    vol_factor = np.ones(T)
    vol_factor[340:450] = 2.5

    # -- Base returns (before injection) --
    garch_scale = np.sqrt(sigma2) * vol_factor[:, None]
    base_returns = Z_corr * garch_scale

    # -- Sector structure --
    sector_indices = [list(range(s * sps, (s + 1) * sps)) for s in range(Ns)]

    # -- Initial injection schedule --
    injection = _build_injection_schedule(T, Ns)
    log_returns = _apply_injection(base_returns, injection, sector_indices, Ns, sps)
    sector_returns = _compute_sector_returns(log_returns, sector_indices, Ns, sps)

    # ================================================================
    # AUTO-CALIBRATION: make |dG0| and lambda overlap in magnitude
    # ================================================================
    c_dg0_cal = C["C_DELTA_G"]
    c_lam_cal = C["C_LAMBDA"]

    # First pass: measure raw ratio and adjust c_lam
    T_eff = T - ROLL
    mom0 = _compute_momentum(sector_returns, ROLL)  # (T_eff, Ns)
    sm0 = np.sort(mom0, axis=1)
    dG0_raw0 = -0.5 * ((sm0[:, -1] + sm0[:, -2]) / 2 - (sm0[:, 0] + sm0[:, 1]) / 2)  # (T_eff,)

    sec_var0 = np.zeros((T, Ns))
    for s in range(Ns):
        sec_var0[:, s] = _rolling_var(sector_returns[:, s], C["ROLL_LAMBDA"])
    lam_raw0 = np.mean(sec_var0[ROLL:], axis=1)  # (T_eff,)

    v0 = np.isfinite(dG0_raw0) & np.isfinite(lam_raw0) & (lam_raw0 > 0)
    med_dG0 = np.median(np.abs(dG0_raw0[v0]))
    med_lam = np.median(lam_raw0[v0])

    c_lam_cal *= (med_dG0 / (med_lam + 1e-20))

    n_inv, ratio, _, _ = _compute_calibration_metrics(
        sector_returns, c_dg0_cal, c_lam_cal, ROLL, Ns)

    print(f"  [CALIBRATION PASS 1] ratio={ratio:.3f}  inverted_days={n_inv}  "
          f"c_dg0={c_dg0_cal:.4f}  c_lam={c_lam_cal:.2f}")

    # Fine-tune ratio into [0.3, 3.0]
    if ratio < 0.3:
        c_lam_cal *= ratio / 0.5
    elif ratio > 3.0:
        c_lam_cal *= ratio / 2.0

    n_inv, ratio, _, _ = _compute_calibration_metrics(
        sector_returns, c_dg0_cal, c_lam_cal, ROLL, Ns)

    # If still insufficient inverted days, boost the injection
    if n_inv < 30:
        boost = 1.0 + 0.5 * (30 - n_inv) / 30.0
        injection[200:340, 0] *= boost
        injection[560:660, 3] *= boost
        injection[560:660, 5] *= boost

        log_returns = _apply_injection(base_returns, injection, sector_indices, Ns, sps)
        sector_returns = _compute_sector_returns(log_returns, sector_indices, Ns, sps)

        n_inv, ratio, _, _ = _compute_calibration_metrics(
            sector_returns, c_dg0_cal, c_lam_cal, ROLL, Ns)
        print(f"  [CALIBRATION PASS 2] boost={boost:.2f}  "
              f"ratio={ratio:.3f}  inverted_days={n_inv}")

    if n_inv < 30:
        boost2 = 1.0 + 1.0 * (30 - n_inv) / 30.0
        injection[200:340, 0] *= boost2
        injection[560:660, 3] *= boost2
        injection[560:660, 5] *= boost2

        log_returns = _apply_injection(base_returns, injection, sector_indices, Ns, sps)
        sector_returns = _compute_sector_returns(log_returns, sector_indices, Ns, sps)

        n_inv, ratio, _, _ = _compute_calibration_metrics(
            sector_returns, c_dg0_cal, c_lam_cal, ROLL, Ns)
        print(f"  [CALIBRATION PASS 3] boost2={boost2:.2f}  "
              f"ratio={ratio:.3f}  inverted_days={n_inv}")

    return log_returns, sector_returns, sector_indices, c_dg0_cal, c_lam_cal


if __name__ == "__main__":
    print("=" * 60)
    print("DATA MODULE -- VERIFICATION")
    print("=" * 60)
    sr, sec_r, si, cdg, clm = generate_returns()
    print(f"  stock_returns shape: {sr.shape}")
    print(f"  sector_returns shape: {sec_r.shape}")
    print(f"  sector_indices: {si}")
    print("  PASS: data.py executes without error")