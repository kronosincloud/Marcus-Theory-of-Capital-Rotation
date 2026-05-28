"""
Marcus Theory of Capital Rotation — Data Generation Module
Generates synthetic Cholesky-correlated GBM returns with GARCH volatility
and a sector momentum injection schedule with TEMPORAL LAG between
spread growth and vol spike — the mechanism that creates the inverted region.
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


def _rolling_mean(x, window):
    n = len(x)
    out = np.full(n, np.nan)
    if n < window:
        return out
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out[window - 1:] = (cs[window:] - cs[:-window]) / window
    return out


def _ease_quintic(x):
    """Quintic ease-in-out for smooth vol ramp transitions."""
    x = np.clip(x, 0.0, 1.0)
    return 6 * x**5 - 15 * x**4 + 10 * x**3


def _build_injection_schedule(T, N_s):
    """
    Returns (T, N_s) array of daily mean log-returns per sector.
    
    KEY DESIGN: The momentum differential runs AHEAD of the vol spike.
    This temporal lag is what creates the inverted region — during the
    gap, spread is large but λ is still at baseline.
    """
    sched = np.zeros((T, N_s))
    # 0=Tech, 1=Fin, 2=Health, 3=Energy, 4=ConsDisc, 5=Indust

    # Phase 1: NORMAL REGION (days 0-119)
    sched[:120, 0] = 0.0012
    sched[:120, 3] = 0.0003

    # Phase 2: APPROACHING ACTIVATIONLESS (days 120-199)
    for t in range(120, 200):
        p = (t - 120) / 80.0
        sched[t, 0] = 0.0012 + p * (0.0030 - 0.0012)
        sched[t, 3] = 0.0003 + p * (-0.0005 - 0.0003)

    # Phase 3: INVERTED WINDOW — momentum spikes, vol still at baseline
    #          (days 200-260)
    sched[200:260, 0] = 0.0045
    sched[200:260, 3] = -0.0005

    # Phase 4: VOL RAMP — vol starts catching up, momentum moderates
    #          (days 260-330)
    for t in range(260, 330):
        p = (t - 260) / 70.0
        sched[t, 0] = 0.0045 - p * (0.0045 - 0.0025)
        sched[t, 3] = -0.0005 + p * (-0.0005 - (-0.0003))

    # Phase 5: CRISIS REBALANCING — full vol, momentum normalizing
    #          (days 330-420)
    sched[330:420, 0] = 0.0010
    sched[330:420, 3] = 0.0010

    # Phase 6: RECOVERY (days 420-510)
    for t in range(420, 510):
        cyc = np.sin(2 * np.pi * (t - 420) / 40.0)
        sched[t, 0] = 0.0015 + 0.0008 * cyc
        sched[t, 3] = 0.0005 - 0.0008 * cyc

    # Phase 7: INVERTED WINDOW 2 — Energy/Industrials expensive
    #          (days 510-570)
    sched[510:570, 3] = 0.0030
    sched[510:570, 5] = 0.0025
    sched[510:570, 0] = -0.0010

    # Phase 8: VOL RAMP 2 (days 570-640)
    for t in range(570, 640):
        p = (t - 570) / 70.0
        sched[t, 3] = 0.0030 - p * (0.0030 - 0.0015)
        sched[t, 5] = 0.0025 - p * (0.0025 - 0.0010)
        sched[t, 0] = -0.0010 + p * (-0.0010 - 0.0015)

    # Phase 9: CRISIS REBALANCING 2 (days 640-720)
    sched[640:720, 0] = 0.0010
    sched[640:720, 3] = 0.0010

    # Phase 10: ACTIVATIONLESS CODA (days 720-756)
    sched[720:756, 0] = 0.0018
    sched[720:756, 3] = 0.0008

    return sched


def _build_vol_factor(T):
    """
    Vol factor with temporal lag. Vol stays at 1.0 during momentum
    spikes, then ramps up AFTER a delay. That delay IS the inverted window.
    """
    vf = np.ones(T)
    crisis = CONFIG["VOL_CRISIS"]  # 3.0
    normal = 1.0

    # Episode 1 ramp: days 260-330 (70 days, linear)
    for t in range(260, 330):
        p = (t - 260) / 70.0
        vf[t] = normal + (crisis - normal) * p
    # Episode 1 hold: days 330-400
    vf[330:400] = crisis
    # Episode 1 decay: days 400-460
    for t in range(400, 460):
        p = (t - 400) / 60.0
        vf[t] = crisis + (normal - crisis) * p

    # Episode 2 ramp: days 570-640 (70 days, linear)
    for t in range(570, 640):
        p = ( t - 570) / 70.0
        vf[t] = normal + (crisis - normal) * p
    # Episode 2 hold: days 640-710
    vf[640:710] = crisis
    # Episode 2 decay: days 710-770
    for t in range(710, min(T, 770)):
        p = (t - 710) / 60.0
        vf[t] = crisis + (normal - crisis) * p

    return vf


def _apply_injection(log_returns, injection, sector_indices, Ns, sps):
    out = log_returns.copy()
    for s in range(Ns):
        out[:, sector_indices[s]] += injection[:, s:s+1]
    return out


def _compute_sector_returns(log_returns, sector_indices, Ns, sps):
    T = log_returns.shape[0]
    sec_ret = np.zeros((T, Ns))
    for s in range(Ns):
        sec_ret[:, s] = np.mean(log_returns[:, sector_indices[s]], axis=1)
    return sec_ret


def _compute_momentum(sector_returns, ROLL):
    T, Ns = sector_returns.shape
    cum = np.vstack([np.zeros((1, Ns)), np.cumsum(sector_returns, axis=0)])
    return cum[ROLL:T] - cum[0:T - ROLL]


def _compute_calibration_metrics(sector_returns, c_dg0, c_lam, ROLL, Ns):
    mom = _compute_momentum(sector_returns, ROLL)
    sorted_mom = np.sort(mom, axis=1)
    m_bot2 = (sorted_mom[:, 0] + sorted_mom[:, 1]) / 2.0
    m_top2 = (sorted_mom[:, -1] + sorted_mom[:, -2]) / 2.0
    dG0_raw = -0.5 * (m_top2 - m_bot2)
    T = sector_returns.shape[0]
    sec_var = np.zeros((T, Ns))
    for s in range(Ns):
        sec_var[:, s] = _rolling_var(sector_returns[:, s], CONFIG["ROLL_LAMBDA"])
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
    The temporal lag between spread growth and vol spike is the
    mechanism that creates the inverted region window.
    """
    C = CONFIG
    rng = np.random.RandomState(C["SEED"])
    N  = C["N_STOCKS"]
    T  = C["T_TOTAL"]
    Ns = C["N_SECTORS"]
    sps = C["STOCKS_PER_SECTOR"]
    ROLL = C["ROLL_MOMENTUM"]

    # -- Correlation matrix --
    corr = np.full((N, N), C["RHO_ACROSS"])
    for s in range(Ns):
        i0, i1 = s * sps, (s + 1) * sps
        corr[i0:i1, i0:i1] = C["RHO_WITHIN"]
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr)

    # -- GARCH(1,1) variance (independent shocks — Trap-4 safe) --
    sigma2_daily = (C["SIGMA_BASE"] ** 2) / 252.0
    garch_shocks = rng.randn(T, N)
    sigma2 = np.full((T, N), sigma2_daily)
    for t in range(1, T):
        sigma2[t] = (C["GARCH_OMEGA"]
                     + C["GARCH_ALPHA"] * garch_shocks[t - 1] ** 2 * sigma2_daily
                     + C["GARCH_BETA"] * sigma2[t - 1])
        sigma2[t] = np.clip(sigma2[t], sigma2_daily * 0.1, sigma2_daily * 100.0)

    # -- Correlated normals --
    Z = rng.randn(T, N)
    Z_corr = Z @ L.T

    # -- Vol factor with temporal lag --
    vol_factor = _build_vol_factor(T)

    # -- Base returns (before injection) --
    garch_scale = np.sqrt(sigma2) * vol_factor[:, None]
    base_returns = Z_corr * garch_scale

    # -- Sector structure --
    sector_indices = [list(range(s * sps, (s + 1) * sps)) for s in range(Ns)]

    # -- Initial injection --
    injection = _build_injection_schedule(T, Ns)
    log_returns = _apply_injection(base_returns, injection, sector_indices, Ns, sps)
    sector_returns = _compute_sector_returns(log_returns, sector_indices, Ns, sps)

    # ================================================================
    # CALIBRATION — c_lam from normal-period variance (NOT proportional to spread)
    #
    # This is the KEY FIX. Previously c_lam was scaled by med_dG0, which
    # forced λ to track |ΔG⁰| lockstep. Now c_lam is a fixed physical
    # quantity derived from the normal-vol baseline. The inverted region
    # is created entirely by the temporal lag between spread and vol.
    # ================================================================
    T_eff = T - ROLL

    # Step 1: Measure normal-period variance (days 50-119, before injection)
    normal_sec_var = np.zeros(Ns)
    for s in range(Ns):
        rv = _rolling_var(sector_returns[50:120, s], CONFIG["ROLL_LAMBDA"])
        normal_sec_var[s] = np.nanmean(rv)
    normal_lam_raw = np.mean(normal_sec_var)

    target_lam = C["TARGET_LAM_NORMAL"]  # 0.12
    c_lam_cal = target_lam / (normal_lam_raw + 1e-20)

    # Step 2: Set initial c_dg0 and measure inverted count
    c_dg0_cal = C["C_DELTA_G"]  # 1.0
    n_inv, ratio, _, _ = _compute_calibration_metrics(
        sector_returns, c_dg0_cal, c_lam_cal, ROLL, Ns)
    print(f"  [CALIBRATION] c_lam={c_lam_cal:.2f}  "
          f"(from normal-period var)  "
          f"inverted_days={n_inv}  c_dg0={c_dg0_cal:.4f}")

    # Step 3: If too few inverted days, boost injection strength
    if n_inv < C["MIN_INV_DAYS"]:
        boost = 1.0 + 0.5 * (C["MIN_INV_DAYS"] - n_inv) / C["MIN_INV_DAYS"]
        injection[200:260, 0] *= boost
        injection[510:570, 3] *= boost
        injection[510:570, 5] *= boost
        log_returns = _apply_injection(base_returns, injection, sector_indices, Ns, sps)
        sector_returns = _compute_sector_returns(log_returns, sector_indices, Ns, sps)
        n_inv, ratio, _, _ = _compute_calibration_metrics(
            sector_returns, c_dg0_cal, c_lam_cal, ROLL, Ns)
        print(f"  [CALIBRATION BOOST] boost={boost:.2f}  inverted_days={n_inv}")

    # Step 4: If too many inverted days, reduce c_dg0 (NOT c_lam!)
    max_inv = C["MAX_INV_DAYS"]  # 70
    target_inv = C["TARGET_INV_DAYS"]  # 55
    if n_inv > max_inv:
        c_dg0_cal *= np.sqrt(target_inv / n_inv)
        n_inv, ratio, _, _ = _compute_calibration_metrics(
            sector_returns, c_dg0_cal, c_lam_cal, ROLL, Ns)
        print(f"  [CALIBRATION DOWNSCALE] c_dg0 -> {c_dg0_cal:.4f}  "
              f"inverted_days: {n_inv}  ratio={ratio:.3f}")

        if n_inv > max_inv:
            c_dg0_cal *= np.sqrt(target_inv / n_inv)
            n_inv_final, ratio_final, _, _ = _compute_calibration_metrics(
                sector_returns, c_dg0_cal, c_lam_cal, ROLL, Ns)
            print(f"  [CALIBRATION DOWNSCALE 2] c_dg0 -> {c_dg0_cal:.4f}  "
                  f"inverted_days: {n_inv_final}")

    # Also verify ratio is in [0.3, 3.0]
    if ratio < 0.3:
        c_dg0_cal *= 0.3 / ratio
        n_inv, ratio, _, _ = _compute_calibration_metrics(
            sector_returns, c_dg0_cal, c_lam_cal, ROLL, Ns)
        print(f"  [CALIBRATION RATIO FIX] c_dg0 -> {c_dg0_cal:.4f}  ratio={ratio:.3f}")
    elif ratio > 3.0:
        c_dg0_cal *= 3.0 / ratio
        n_inv, ratio, _, _ = _compute_calibration_metrics(
            sector_returns, c_dg0_cal, c_lam_cal, ROLL, Ns)
        print(f"  [CALIBRATION RATIO FIX] c_dg0 -> {c_dg0_cal:.4f}  ratio={ratio:.3f}")

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