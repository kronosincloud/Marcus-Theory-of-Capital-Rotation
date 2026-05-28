"""
Marcus Theory of Capital Rotation — Static Visualisation Module
Renders the 1920x1080 Bloomberg Dark dashboard PNG.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

from config import CONFIG, THEME, CMAP_MARCUS
from engine import (
    compute_marcus_rate_normalized,
    compute_2d_marcus_map,
)


def _style_ax(ax):
    ax.set_facecolor(THEME["PANEL_BG"])
    for sp in ax.spines.values():
        sp.set_color(THEME["SPINE"])
        sp.set_linewidth(0.5)
    ax.tick_params(colors=THEME["TEXT_DIM"], labelsize=7,
                   direction="in", length=3)
    ax.yaxis.grid(True, color=THEME["GRID"], lw=0.3, alpha=0.4)


def render_png(data):
    dG0_v   = data["dG0_valid"]
    lam_v   = data["lam_valid"]
    kBT_v   = data["kBT_valid"]
    k_act   = data["k_actual_valid"]
    k_mar   = data["k_marcus_valid"]
    dG_dd   = data["dG_barrier"]
    t_v     = data["t_valid"]
    K_surf  = data["K_surface"]
    DG0_m   = data["DG0_mesh"]
    T_m     = data["T_mesh"]
    dG0_g   = data["dG0_grid"]
    lam_max = data["lam_max"]
    kBT_mn  = data["kBT_mean"]
    inv_m   = data["inverted_mask"]

    T_eff = len(t_v)
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(CONFIG["FIG_W"], CONFIG["FIG_H"]),
                     dpi=CONFIG["DPI_PNG"], facecolor=THEME["BG"])
    fig.patch.set_facecolor(THEME["BG"])

    gs = GridSpec(4, 2, width_ratios=[2.2, 1],
                  left=0.06, right=0.97, top=0.87, bottom=0.07,
                  hspace=0.38, wspace=0.10, figure=fig)

    # ═══════════ 3D SURFACE ═══════════
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.set_facecolor(THEME["BG"])

    ax3d.plot_surface(
        DG0_m, T_m, K_surf,
        cmap=CMAP_MARCUS, norm=norm,
        alpha=0.90, rstride=1, cstride=1,
        edgecolor=(0.0, 0.95, 1.0, 0.08), linewidth=0.22,
        antialiased=True, zorder=1,
    )

    z_floor = -0.04
    ax3d.contourf(DG0_m, T_m, K_surf,
                  zdir="z", offset=z_floor,
                  cmap=CMAP_MARCUS, norm=norm, alpha=0.18, levels=12)

    dG0_ridge = -lam_v
    ax3d.plot(dG0_ridge, t_v, np.ones(T_eff),
              color=THEME["ORANGE"], lw=3.2, alpha=0.95, zorder=15)
    ax3d.scatter([dG0_ridge[-1]], [t_v[-1]], [1.0],
                 s=36, color=THEME["YELLOW"], edgecolor="white",
                 linewidth=0.5, zorder=20)

    k_on_surf = compute_marcus_rate_normalized(dG0_v, lam_v, kBT_v)
    ax3d.plot(dG0_v, t_v, k_on_surf,
              color=THEME["CYAN"], lw=2.0, alpha=0.85, zorder=14)

    if np.any(inv_m):
        ax3d.scatter(
            dG0_v[inv_m], t_v[inv_m], k_on_surf[inv_m] + 0.03,
            s=40, color=THEME["YELLOW"], marker="*", zorder=18,
            edgecolor="white", linewidths=0.4,
        )

    pane_color = (0.02, 0.02, 0.02, 1.0)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.set_pane_color(pane_color)
        axis._axinfo["grid"]["color"] = (0.13, 0.13, 0.13, 0.5)
        axis._axinfo["grid"]["linewidth"] = 0.35

    # FIX: Shortened labels to prevent cutoff at azim=-50
    ax3d.set_xlabel(r"$\Delta G^0$",
                    fontsize=12, fontweight="bold",
                    color=THEME["TEXT_DIM"], labelpad=12,
                    fontfamily=THEME["FONT"])
    ax3d.set_ylabel("TIME [days]",
                    fontsize=11, fontweight="bold",
                    color=THEME["TEXT_DIM"], labelpad=12,
                    fontfamily=THEME["FONT"])
    ax3d.set_zlabel(r"$\hat{k}$  ROTATION RATE",
                    fontsize=12, fontweight="bold",
                    color=THEME["TEXT_DIM"], labelpad=12,
                    fontfamily=THEME["FONT"])
    ax3d.set_zlim(-0.04, 1.10)
    ax3d.set_box_aspect([1.6, 2.0, 0.75])
    ax3d.view_init(elev=30, azim=-50)
    ax3d.tick_params(axis="both", colors=THEME["TEXT_DIM"], labelsize=8)

    # ═══════════ BOTTOM-LEFT INSET — 2D Marcus Map ═══════════
    # FIX: Adjusted position and sizing to prevent label cutoff
    ax_ins = fig.add_axes([0.065, 0.08, 0.19, 0.13], zorder=100)
    ax_ins.set_facecolor("#0a0a0a")
    for sp in ax_ins.spines.values():
        sp.set_color(THEME["SPINE"])
        sp.set_linewidth(0.5)
    ax_ins.tick_params(axis="both", colors=THEME["TEXT_DIM"],
                       labelsize=6, direction="in")

    dG0_2d = np.linspace(-4 * lam_max, lam_max, CONFIG["N_DG0_GRID"])
    lam_2d = np.linspace(0.001, 3 * lam_max, CONFIG["N_LAM_GRID"])
    K2D = compute_2d_marcus_map(dG0_2d, lam_2d, kBT_mn)

    ax_ins.imshow(K2D.T, origin="lower", aspect="auto",
                  cmap=CMAP_MARCUS, vmin=0, vmax=1,
                  extent=[dG0_2d[0], dG0_2d[-1], lam_2d[0], lam_2d[-1]])

    diag_lam = np.linspace(lam_2d[0], lam_2d[-1], 200)
    ax_ins.plot(-diag_lam, diag_lam,
                color=THEME["ORANGE"], lw=1.2, ls="--", alpha=0.8)

    pts = np.array([dG0_v, lam_v]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="cool", linewidth=1.4, alpha=0.90)
    lc.set_array(np.linspace(0, 1, len(dG0_v) - 1))
    lc.set_clim(0, 1)
    ax_ins.add_collection(lc)

    ax_ins.set_xlabel(r"$\Delta G^0$", fontsize=6,
                      color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])
    ax_ins.set_ylabel(r"$\lambda$", fontsize=6,
                      color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])
    ax_ins.set_title("2D MARCUS MAP", fontsize=7,
                     color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])

    # ═══════════ RIGHT PANELS ═══════════
    panels = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    for p in panels:
        _style_ax(p)

    # -- P1: Driving Force --
    ax1 = panels[0]
    ax1.plot(t_v, dG0_v, color=THEME["CYAN"], lw=1.0, label=r"$\Delta G^0$")
    ax1.plot(t_v, -lam_v, color=THEME["ORANGE"], lw=1.0, ls="--",
             label=r"$-\lambda$")
    ax1.fill_between(t_v, dG0_v, -lam_v,
                     where=(dG0_v < -lam_v),
                     color=THEME["RED"], alpha=0.15)
    ax1.set_ylabel(r"$\Delta G^0$", color=THEME["TEXT_DIM"],
                   fontsize=9, fontfamily=THEME["FONT"])
    ax1.set_title("DRIVING FORCE  " + r"$\Delta G^0(t)$",
                  fontsize=9, color=THEME["TEXT_DIM"],
                  fontfamily=THEME["FONT"], pad=4)
    leg1 = ax1.legend(loc="upper left", fontsize=7,
                      facecolor=THEME["BG"], edgecolor=THEME["GRID"])
    for txt in leg1.get_texts():
        txt.set_color(THEME["TEXT_DIM"])

    # -- P2: Reorganisation Energy --
    ax2 = panels[1]
    ax2.plot(t_v, lam_v, color=THEME["YELLOW"], lw=1.0, label=r"$\lambda$")
    ax2.plot(t_v, kBT_v, color=THEME["GREEN"], lw=1.0, alpha=0.7,
             label=r"$k_BT$")
    ax2.set_ylabel(r"$\lambda$", color=THEME["TEXT_DIM"],
                   fontsize=9, fontfamily=THEME["FONT"])
    ax2.set_title("REORGANIZATION ENERGY  " + r"$\lambda(t)$",
                  fontsize=9, color=THEME["TEXT_DIM"],
                  fontfamily=THEME["FONT"], pad=4)
    leg2 = ax2.legend(loc="upper left", fontsize=7,
                      facecolor=THEME["BG"], edgecolor=THEME["GRID"])
    for txt in leg2.get_texts():
        txt.set_color(THEME["TEXT_DIM"])

    # -- P3: Activation Barrier --
    ax3 = panels[2]
    ax3.plot(t_v, dG_dd, color=THEME["MAGENTA"], lw=1.0)
    ax3.fill_between(t_v, 0, dG_dd, color=THEME["MAGENTA"], alpha=0.20)
    ax3.axhline(0, color="#ffffff", lw=0.3, alpha=0.3)
    ax3.set_ylabel(r"$\Delta G^\ddagger$", color=THEME["TEXT_DIM"],
                   fontsize=9, fontfamily=THEME["FONT"])
    ax3.set_title("ACTIVATION BARRIER  " + r"$\Delta G^\ddagger(t)$",
                  fontsize=9, color=THEME["TEXT_DIM"],
                  fontfamily=THEME["FONT"], pad=4)

    # -- P4: Marcus vs Actual --
    ax4 = panels[3]
    valid_corr = np.isfinite(k_act) & np.isfinite(k_mar)
    corr_val = np.corrcoef(k_mar[valid_corr], k_act[valid_corr])[0, 1]
    ax4.plot(t_v, k_mar, color=THEME["ORANGE"], lw=1.0,
             label=r"$k_{Marcus}$")
    ax4.plot(t_v, k_act, color=THEME["CYAN"], lw=1.0,
             label=r"$k_{actual}$")
    ax4.set_ylim(-0.05, 1.10)
    ax4.set_xlabel("Time [days]", color=THEME["TEXT_DIM"],
                   fontsize=9, fontfamily=THEME["FONT"])
    ax4.set_ylabel(r"$\hat{k}$", color=THEME["TEXT_DIM"],
                   fontsize=9, fontfamily=THEME["FONT"])
    ax4.set_title(r"$k_{Marcus}$ (ORANGE)  vs  $k_{actual}$ (CYAN)",
                  fontsize=9, color=THEME["TEXT_DIM"],
                  fontfamily=THEME["FONT"], pad=4)
    ax4.text(0.97, 0.92, f"corr = {corr_val:.3f}",
             transform=ax4.transAxes, ha="right", va="top",
             fontsize=9, color=THEME["YELLOW"], fontfamily=THEME["FONT"],
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d0d0d",
                       edgecolor=THEME["GRID"], alpha=0.8))
    leg4 = ax4.legend(loc="upper left", fontsize=7,
                      facecolor=THEME["BG"], edgecolor=THEME["GRID"])
    for txt in leg4.get_texts():
        txt.set_color(THEME["TEXT_DIM"])

    # ═══════════ TITLE BLOCK ═══════════
    n_inv = int(np.sum(inv_m))
    lam_mean = float(np.mean(lam_v))

    fig.text(0.50, 0.960,
             "MARCUS THEORY OF CAPITAL ROTATION  —  INVERTED REGION DYNAMICS",
             ha="center", va="center",
             fontsize=24, fontweight="bold",
             color=THEME["ORANGE"], fontfamily=THEME["FONT"])

    fig.text(0.50, 0.932,
             f"$\\hat{{k}}(\\Delta G^0,t)=\\exp(-(\\Delta G^0+\\lambda)^2/4\\lambda k_BT)$"
             f"     $\\lambda_{{mean}}={lam_mean:.4f}$"
             f"     $k_BT_{{mean}}={kBT_mn:.4f}$"
             f"     INVERTED EVENTS: {n_inv:d}",
             ha="center", va="center",
             fontsize=10, color=THEME["TEXT_DIM"],
             fontfamily=THEME["FONT"])

    psi = -dG0_v[-1] - lam_v[-1]
    if psi > 0.5 * lam_v[-1]:
        phase_str = "DEEP INVERTED"
    elif psi > 0:
        phase_str = "INVERTED REGION"
    elif abs(psi) <= 0.05 * lam_v[-1]:
        phase_str = "ACTIVATIONLESS"
    else:
        phase_str = "NORMAL REGION"

    fig.text(0.96, 0.900,
             f"$\\Delta G^0$={dG0_v[-1]:.4f}    "
             f"$\\lambda$={lam_v[-1]:.4f}    "
             f"$\\Delta G^{{\\ddagger}}$={dG_dd[-1]:.5f}    "
             f"PHASE: {phase_str}",
             ha="right", va="center",
             fontsize=10, fontweight="bold",
             color=THEME["YELLOW"], fontfamily=THEME["FONT"])

    fig.text(0.985, 0.010, CONFIG["WATERMARK"],
             ha="right", va="bottom", fontsize=10,
             color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"], alpha=0.6)

    fig.savefig(CONFIG["OUT_PNG"], dpi=CONFIG["DPI_PNG"],
                facecolor=THEME["BG"])
    plt.close(fig)
    print(f"  [VISUAL] PNG saved -> {CONFIG['OUT_PNG']}")