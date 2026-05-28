"""
Marcus Theory of Capital Rotation — Computation Engine
All Marcus rate, barrier, and diagnostic calculations.
Every function uses the mandatory 5-part docstring.
"""

import numpy as np
from config import CONFIG


def compute_sector_momentum(sector_returns, roll_window):
    """
    CHEMICAL ANALOGY:
        In electron transfer, the donor and acceptor each occupy a minimum on
        a potential energy parabola. The "position" of each species along the
        reaction coordinate is determined by its nuclear configuration. Here,
        the sector momentum is the financial analog of that position — it tells
        us how far each sector has moved from its equilibrium along the
        valuation reaction coordinate.

    FINANCIAL INTERPRETATION:
        m_s(t) is the 252-day rolling total log-return of sector s. It measures
        the cumulative performance over approximately one year. Sectors with
        high positive momentum are the "donors" (overvalued, high energy);
        sectors with low or negative momentum are the "acceptors" (undervalued,
        low energy). This is the raw input to the driving-force calculation.

    MATHEMATICAL FORMULA:
        m_s(t) = sum_{tau=t-W}^{t-1} r_s(tau)
        where W = ROLL_MOMENTUM (252), r_s(tau) is the daily log-return of
        the equal-weighted sector-s portfolio, and the sum uses the cumulative-
        sum trick for O(T) efficiency.

    NUMERICAL STABILITY NOTES:
        - Uses cumulative-sum differencing, not a rolling loop, for speed.
        - FIX: cum[W:T] - cum[0:T-W] yields exactly T-W values for days W..T-1.
          The naive cum[W:] - cum[:-W] would give T-W+1 values including a
          phantom day T, causing shape mismatches downstream.
        - Returns a (T-W, N_s) array with no NaN values.

    EXPECTED OUTPUT RANGES:
        Shape: (T - ROLL_MOMENTUM, N_SECTORS). Typical values: -0.5 to +0.8
        for the injection schedule used. May exceed +/-1.0 during extreme
        inverted-region episodes.
    """
    T, Ns = sector_returns.shape
    W = roll_window
    cum = np.vstack([np.zeros((1, Ns)), np.cumsum(sector_returns, axis=0)])  # (T+1, Ns)
    # FIX: Use cum[W:T] - cum[0:T-W] to get exactly T-W values (days W to T-1)
    momentum = cum[W:T] - cum[0:T - W]
    return momentum


def compute_driving_force(momentum_valid, c_dg0, n_top_bot):
    """
    CHEMICAL ANALOGY:
        Delta G0 is the standard Gibbs free-energy change of the electron transfer
        reaction. A negative Delta G0 means the reaction is thermodynamically
        favorable: the electron lowers its energy by moving from donor to
        acceptor. In Marcus theory, this is the vertical offset between the
        two parabola minima on the free-energy diagram.

    FINANCIAL INTERPRETATION:
        Delta G0(t) measures the signed valuation spread between the most expensive
        (top-momentum) and cheapest (bottom-momentum) sector clusters. When
        Delta G0 is very negative, expensive sectors dominate — there is a large
        thermodynamic incentive for capital to rotate toward cheap sectors.
        The sign convention mirrors chemistry: negative Delta G0 = favorable rotation.

    MATHEMATICAL FORMULA:
        Delta G0(t) = -(c_DG / 2) * [m_top2(t) - m_bot2(t)]
        where m_top2 = (m_(N) + m_(N-1)) / 2  (two largest momenta)
              m_bot2 = (m_(1) + m_(2)) / 2    (two smallest momenta)
        and m_(k) is the k-th order statistic of the sector momenta at time t.

    NUMERICAL STABILITY NOTES:
        - Sorting is O(N_s log N_s) per timestep; with N_s=6 this is trivial.
        - c_dg0 is the auto-calibrated scaling constant from data.py that
          brings Delta G0 into the same numerical range as lambda.
        - No division by zero possible.

    EXPECTED OUTPUT RANGES:
        Shape: (T_valid,). Typical range: -0.5 to +0.1. Negative values
        dominate when expensive sectors outperform. Extreme inverted episodes
        may push below -0.3.
    """
    sorted_m = np.sort(momentum_valid, axis=1)
    m_bot2 = (sorted_m[:, 0] + sorted_m[:, 1]) / 2.0
    m_top2 = (sorted_m[:, -1] + sorted_m[:, -2]) / 2.0
    dG0 = -0.5 * (m_top2 - m_bot2) * c_dg0
    return dG0


def compute_reorganization_energy(sector_returns_full, roll_window, c_lambda):
    """
    CHEMICAL ANALOGY:
        lambda is the reorganization energy: the energy cost of distorting the
        nuclear framework of both donor and acceptor from their equilibrium
        geometries to the transition-state geometry. It is the "friction" of
        the reaction — the cost of structural rearrangement that must be paid
        before the electron can jump. In solvent-mediated electron transfer,
        lambda reflects the solvent polarizability and molecular size.

    FINANCIAL INTERPRETATION:
        lambda(t) represents the total market friction that impedes capital rotation:
        bid-ask spreads, realized volatility, redemption pressure, career risk,
        and portfolio restructuring costs. When volatility spikes (crisis),
        lambda increases — it becomes more expensive to restructure portfolios,
        paradoxically making it HARDER to rotate even when the incentive is
        largest. This is the mechanism behind the inverted region.

    MATHEMATICAL FORMULA:
        lambda(t) = c_lam * (1/N_s) * sum_{s=1}^{N_s} sigma2_s(t)
        where sigma2_s(t) = Var_W(r_s) is the W-day rolling variance of sector s's
        daily return, and c_lam is the auto-calibrated scaling constant.

    NUMERICAL STABILITY NOTES:
        - Rolling variance uses the cumulative-sum trick for O(T) efficiency.
        - Variance is clipped to >= 0 to prevent tiny negative values from
          floating-point arithmetic.
        - Uses axis=1 to average ACROSS sectors, producing a time series.
        - sec_var[ROLL:] has shape (T-ROLL, Ns); mean(axis=1) gives (T-ROLL,).

    EXPECTED OUTPUT RANGES:
        Shape: (T_valid,). Positive always. Typical range: 0.001 to 0.5
        (after calibration). Spikes to 2-5x normal during crisis vol episodes.
    """
    T, Ns = sector_returns_full.shape
    W = roll_window
    sec_var = np.zeros((T, Ns))
    for s in range(Ns):
        x = sector_returns_full[:, s]
        cs  = np.concatenate([[0.0], np.cumsum(x)])
        cs2 = np.concatenate([[0.0], np.cumsum(x ** 2)])
        s1 = cs[W:] - cs[:-W]
        s2 = cs2[W:] - cs2[:-W]
        v = s2 / W - (s1 / W) ** 2
        sec_var[W - 1:, s] = np.maximum(v, 0.0)

    # axis=1 averages ACROSS sectors (columns), yielding shape (T-ROLL,)
    lam = np.mean(sec_var[CONFIG["ROLL_MOMENTUM"]:], axis=1) * c_lambda
    return lam


def compute_thermal_energy(stock_returns, roll_window):
    """
    CHEMICAL ANALOGY:
        k_B T is the thermal energy that determines the width of the Boltzmann
        distribution of nuclear configurations. Higher temperature means the
        system samples a broader range of geometries, making it more likely to
        find the transition state. In the Marcus rate, k_B T appears in the
        denominator of the exponent, so higher T -> broader rate distribution
        (more sensitive to driving force).

    FINANCIAL INTERPRETATION:
        k_B T(t) is the market-level noise: the variance of the equal-weighted
        market return over a short window. High market noise means individual
        sector moves are obscured by overall market movement, broadening the
        effective rate distribution. Low noise means the rate surface is sharply
        peaked around the activationless condition.

    MATHEMATICAL FORMULA:
        k_B T(t) = Var_W(r_M(t))
        where r_M(t) = (1/N) * sum_i r_i(t) is the equal-weighted market return.

    NUMERICAL STABILITY NOTES:
        - Clipped to >= 1e-10 to prevent division by zero in the Marcus exponent.
        - Uses cumulative-sum trick.
        - Sliced to produce exactly (T - ROLL_MOMENTUM,) values, aligned with
          the other valid-period series (momentum, lambda).

    EXPECTED OUTPUT RANGES:
        Shape: (T_valid,). Positive always. Typical: 1e-5 to 5e-4 (daily
        variance units). Spikes during crisis.
    """
    r_M = np.mean(stock_returns, axis=1)
    T = len(r_M)
    W = roll_window
    cs  = np.concatenate([[0.0], np.cumsum(r_M)])
    cs2 = np.concatenate([[0.0], np.cumsum(r_M ** 2)])
    s1 = cs[W:] - cs[:-W]
    s2 = cs2[W:] - cs2[:-W]
    var_M = s2 / W - (s1 / W) ** 2
    var_M = np.maximum(var_M, 1e-10)

    # Align: var_M[i] covers r_M[i : i+W], ending at day i+W-1.
    # We want days ROLL to T-1, so i+W-1 = t => i = t-W+1.
    # For t=ROLL: i = ROLL-W+1. For t=T-1: i = T-W.
    # Slice: var_M[ROLL-W+1 : T-W+1] which has length T-ROLL.
    start_idx = CONFIG["ROLL_MOMENTUM"] - roll_window + 1
    end_idx = T - roll_window + 1
    return var_M[start_idx:end_idx]


def compute_capital_rotation_velocity(momentum_valid, roll_window):
    """
    CHEMICAL ANALOGY:
        This is the empirical observable that the Marcus rate predicts: the
        actual speed at which electrons hop between donor and acceptor. In
        experiments, this is measured via time-resolved spectroscopy. Here,
        we measure it via portfolio weight changes.

    FINANCIAL INTERPRETATION:
        k_actual(t) measures how rapidly capital is rotating between sectors.
        It is computed from the L1-norm of 5-day changes in momentum-normalized
        sector weights. When sectors are rebalancing rapidly (high turnover),
        k_actual approx 1. When capital is frozen (no rebalancing despite valuation
        signals), k_actual approx 0. This is the quantity the Marcus theory should
        predict.

    MATHEMATICAL FORMULA:
        w_s(t) = m_s(t) / sum_s |m_s(t)|
        Delta w_s(t) = w_s(t) - w_s(t - Delta)
        k_actual(t) = sum_s |Delta w_s(t)| / (max_s |Delta w_s(t)| + eps)
        Normalized to [0, 1] via min-max scaling.

    NUMERICAL STABILITY NOTES:
        - eps = 1e-10 prevents division by zero when all weight changes are zero.
        - First (roll_window) values are NaN (insufficient history for Delta w).
        - Min-max normalization handles the raw range [1, N_s] -> [0, 1].

    EXPECTED OUTPUT RANGES:
        Shape: (T_valid,). Values in [0, 1] after normalization. First
        roll_window entries are NaN.
    """
    W = roll_window
    abs_mom = np.abs(momentum_valid)
    w_sums = abs_mom.sum(axis=1, keepdims=True) + 1e-10
    weights = momentum_valid / w_sums
    if len(weights) <= W:
        return np.full(len(weights), np.nan)
    dw = weights[W:] - weights[:-W]
    max_dw = np.max(np.abs(dw), axis=1) + 1e-10
    raw = np.sum(np.abs(dw), axis=1) / max_dw
    raw_min, raw_max = raw.min(), raw.max()
    normed = (raw - raw_min) / (raw_max - raw_min + 1e-10)
    full = np.full(len(momentum_valid), np.nan)
    full[W:] = normed
    return full


def compute_activation_barrier(dG0_series, lam_series):
    """
    CHEMICAL ANALOGY:
        In Marcus electron transfer, Delta G dagger is the free energy height of the crossing
        point between the donor and acceptor parabolas. A molecule must supply this
        energy thermally to reach the transition state geometry q dagger where both charge
        states are equally probable. It is the literal geometric cost of bringing
        the nuclear framework to the intersection of two potential energy parabolas.

    FINANCIAL INTERPRETATION:
        Delta G dagger(t) is the "effective friction cost" of capital rotation at time t. It is
        not simply lambda(t) (total reorganization energy) but the scaled squared deviation
        from the activationless condition. When Delta G dagger is low, rotation is nearly costless.
        When Delta G dagger is high — either because the spread is too small (normal, underused
        opportunity) or too large (inverted, paralysis) — capital rotation is impeded.
        Crucially, Delta G dagger rises in BOTH directions away from the activationless point,
        explaining why both "not enough spread" AND "too much spread" reduce rotation.

    MATHEMATICAL FORMULA:
        Delta G dagger(t) = (Delta G0(t) + lambda(t))^2 / (4*lambda(t))
        Derivation: the donor and acceptor parabolas G_D(q) = 1/2*kappa*q^2 and
        G_A(q) = 1/2*kappa*(q-d)^2 + Delta G0 cross at q dagger = (d/2)(1 + Delta G0/lambda).
        Substituting back: Delta G dagger = G_D(q dagger) = (Delta G0 + lambda)^2/(4*lambda).

    NUMERICAL STABILITY NOTES:
        - When lambda(t) approx 0 (no friction), Delta G dagger -> inf. Guard with np.maximum(lam, 1e-8).
        - Both dG0_series and lam_series must have the same sign convention:
          Delta G0 < 0 means the reaction is thermodynamically favorable.
        - Delta G dagger cannot be negative by construction (squared numerator). If negative
          values appear, there is a unit mismatch between dG0 and lam.

    EXPECTED OUTPUT RANGES:
        Delta G dagger in [0, ~5*lambda_max]. Normal region: Delta G dagger approx lambda/4 at Delta G0=0,
        decreasing to 0 at activationless. Inverted: rising back to lambda/4 at Delta G0=-2*lambda
        and continuing to grow beyond. Shape: (T,). Never negative.
    """
    lam_safe = np.maximum(lam_series, 1e-8)
    return (dG0_series + lam_safe) ** 2 / (4.0 * lam_safe)


def compute_marcus_rate_normalized(dG0, lam, kBT):
    """
    CHEMICAL ANALOGY:
        The Marcus rate constant k_ET gives the probability per unit time that
        an electron transfers from donor to acceptor. The normalized form h_hat
        removes the electronic-coupling prefactor, isolating the nuclear-
        tunnelling contribution. It equals 1 at the activationless condition
        (every thermal fluctuation leads to transfer) and decays as a Gaussian
        away from that condition.

    FINANCIAL INTERPRETATION:
        h_hat(Delta G0, lambda, k_BT) is the predicted capital rotation efficiency, scaled
        so that the maximum possible efficiency equals 1. At the activationless
        point (spread exactly equals friction), rotation is maximally efficient.
        On either side, efficiency drops — slowly in the normal region, rapidly
        in the inverted region. This is the core prediction to be tested against
        the empirical k_actual.

    MATHEMATICAL FORMULA:
        h_hat(Delta G0, lambda, k_BT) = exp( -(Delta G0 + lambda)^2 / (4*lambda*k_BT) )
        This is the Arrhenius factor exp(-Delta G dagger/k_BT) with
        Delta G dagger = (Delta G0 + lambda)^2/(4*lambda).

    NUMERICAL STABILITY NOTES:
        - lambda must be > 0; guard with np.maximum(lam, 1e-8).
        - kBT must be > 0; guard with np.maximum(kBT, 1e-10).
        - The exponent is always <= 0, so the output is always in (0, 1].
        - For very large |Delta G0 + lambda| or very small kBT, the exponent can
          underflow to -inf, giving h_hat = 0. This is correct behavior.

    EXPECTED OUTPUT RANGES:
        Scalar or array in (0, 1]. Exactly 1.0 when Delta G0 = -lambda. Approaches 0
        deep in either the normal or inverted region.
    """
    lam_safe = np.maximum(lam, 1e-8)
    kBT_safe = np.maximum(kBT, 1e-10)
    exponent = -((dG0 + lam_safe) ** 2) / (4.0 * lam_safe * kBT_safe)
    return np.exp(exponent)


def compute_marcus_rate_series(dG0_series, lam_series, kBT_series):
    """
    CHEMICAL ANALOGY:
        Evaluating the Marcus rate along a trajectory of thermodynamic states
        produces a time-series prediction for the transfer rate. In ultrafast
        spectroscopy, this corresponds to measuring the electron transfer rate
        as the solvent configuration evolves. Peaks in the rate correspond to
        solvent fluctuations that momentarily satisfy the activationless
        condition.

    FINANCIAL INTERPRETATION:
        k_Marcus(t) = h_hat(Delta G0(t), lambda(t), k_BT(t)) is the day-by-day Marcus
        prediction for capital rotation efficiency. It should correlate with
        the empirical k_actual(t). Crucially, it predicts the DROP in rotation
        during inverted-region episodes — the periods when the valuation spread
        is so extreme that capital freezes despite the apparent opportunity.

    MATHEMATICAL FORMULA:
        k_Marcus(t) = exp( -(Delta G0(t) + lambda(t))^2 / (4*lambda(t)*k_BT(t)) )
        Applied element-wise to the three input time series.

    NUMERICAL STABILITY NOTES:
        - All three series must have the same length and alignment.
        - NaN in any input propagates to NaN in output (correct behavior for
          the initial padding of k_actual).
        - Guards on lambda and kBT are inside compute_marcus_rate_normalized.

    EXPECTED OUTPUT RANGES:
        Shape: (T_valid,). Values in (0, 1]. Should show clear drops during
        inverted-region episodes and peaks near the activationless condition.
    """
    return compute_marcus_rate_normalized(dG0_series, lam_series, kBT_series)


def compute_2d_marcus_map(dG0_grid, lam_grid, kBT_fixed):
    """
    CHEMICAL ANALOGY:
        The 2D Marcus rate map k(Delta G0, lambda) at fixed temperature is the
        fundamental phase diagram of electron transfer. It shows the three
        regimes (normal, activationless, inverted) as regions in the
        (driving-force, reorganization-energy) plane. The activationless
        diagonal Delta G0 = -lambda is the ridge of maximum rate. This map is the
        analog of a phase diagram in thermodynamics.

    FINANCIAL INTERPRETATION:
        The 2D map is the "phase space" of capital rotation. The market's
        trajectory (Delta G0(t), lambda(t)) is overlaid on this map, showing which
        regime the market occupies at each moment. The diagonal ridge is the
        optimal rebalancing line. When the trajectory crosses into the region
        below-left of the ridge (inverted), the market is in a state of
        "extreme valuation paralysis."

    MATHEMATICAL FORMULA:
        K2D(i,j) = exp( -(Delta G0_i + lambda_j)^2 / (4*lambda_j*k_BT_fixed) )
        Computed over the full (Delta G0, lambda) grid at a single fixed k_BT value
        (the time-averaged thermal energy).

    NUMERICAL STABILITY NOTES:
        - lam_grid must start above 0 (use 0.001 minimum) to avoid division
          by zero in the exponent denominator.
        - kBT_fixed must be > 0.
        - The result is a 2D array of shape (len(dG0_grid), len(lam_grid)).
        - Fully vectorized via broadcasting; no loops needed.

    EXPECTED OUTPUT RANGES:
        Shape: (N_DG0_GRID, N_LAM_GRID). Values in (0, 1]. The ridge
        (diagonal Delta G0 = -lambda) has value exactly 1.0. Both corners far from
        the ridge approach 0.
    """
    lam_grid_safe = np.maximum(lam_grid, 1e-8)
    kBT_safe = max(kBT_fixed, 1e-10)
    exponent = (-(dG0_grid[:, None] + lam_grid_safe[None, :]) ** 2
                / (4.0 * lam_grid_safe[None, :] * kBT_safe))
    return np.exp(exponent)