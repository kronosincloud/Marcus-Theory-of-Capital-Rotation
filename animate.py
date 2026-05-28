"""
Marcus Theory of Capital Rotation — Animation Module
120-frame GIF at 10 FPS (12-second loop).
GROW -> HOLD -> ORBIT phases.
Mirror Image Rule: identical layout, norm, and title block as visual.py.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import imageio

from config import CONFIG, THEME, CMAP_MARCUS
from engine import compute_marcus_rate_normalized, compute_2d_marcus_map


def _canvas_to_rgb(fig):
    fig.canvas.draw()
    try:
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    except AttributeError:
        w, h = fig.canvas.get_width_height()
        return np.frombuffer(fig.canvas.tostring_rgb(),
                             dtype=np.uint8).reshape(h, w, 3)


def _style_ax(ax):
    ax.set_facecolor(THEME["PANEL_BG"])
    for sp in ax.spines.values():
        sp.set_color(THEME["SPINE"])
        sp.set_linewidth(0.5)
    ax.tick_params(colors=THEME["TEXT_DIM"], labelsize=7,
                   direction="in", length=3)
    ax.yaxis.grid(True, color=THEME["GRID"], lw=0.3, alpha=0.4)


def _ease_quintic(x):
    x = np.clip(x, 0.0, 1.0)
    return 6 * x**5 - 15 * x**4 + 10 * x**3


def _determine_phase(dG0_now, lam_now):
    psi = -dG0_now - lam_now
    if psi > 0.5 * lam_now:
        return "DEEP INVERTED"
    elif psi > 0:
        return "INVERTED REGION"
    elif abs(psi) <= 0.05 * lam_now:
        return "ACTIVATIONLESS"
    else:
        return "NORMAL REGION"


def render_gif(data):
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

    dG0_2d = np.linspace(-4 * lam_max, lam_max, CONFIG["N_DG0_GRID"])
    lam_2d = np.linspace(0.001, 3 * lam_max, CONFIG["N_LAM_GRID"])
    K2D = compute_2d_marcus_map(dG0_2d, lam_2d, kBT_mn)

    k_on_surf = compute_marcus_rate_normalized(dG0_v, lam_v, kBT_v)
    dG0_ridge = -lam_v

    lam_mean = float(np.mean(lam_v))

    N_grow, N_hold, N_orbit = 45, 20, 55
    schedule = []

    for i in range(N_grow):
        raw = i / max(1, N_grow - 1)
        eased = _ease_quintic(raw)
        tc = max(2, int(eased * T_eff))
        schedule.append({
            "phase": "GROW", "tc": tc,
            "z_scale": 0.05 + 0.95 * eased,
            "elev": 8 + 22 * eased,
            "azim": -70 + 20 * eased,
        })

    for i in range(N_hold):
        schedule.append({
            "phase": "HOLD", "tc": T_eff, "z_scale": 1.0,
            "elev": 30 + 2 * np.sin(2 * np.pi * i / N_hold),
            "azim": -50 + 6 * (i / N_hold),
        })

    hold_end_azim = -50 + 6 * ((N_hold - 1) / N_hold)
    hold_end_elev = 30 + 2 * np.sin(2 * np.pi * (N_hold - 1) / N_hold)
    for orb_prog in np.linspace(0.0, 1.0, N_orbit):
        schedule.append({
            "phase": "ORBIT", "tc": T_eff, "z_scale": 1.0,
            "elev": hold_end_elev + 20 * np.sin(np.pi * orb_prog * 1.3),
            "azim": hold_end_azim + 360.0 * orb_prog,
        })

    total_frames = len(schedule)
    print(f"  [ANIMATE] Rendering {total_frames} frames ...")

    frames = []
    for fi, sched in enumerate(schedule):
        tc = sched["tc"]
        zs = sched["z_scale"]

        fig = plt.figure(figsize=(CONFIG["FIG_W"], CONFIG["FIG_H"]),
                         dpi=CONFIG["DPI_GIF"], facecolor=THEME["BG"])
        fig.patch.set_facecolor(THEME["BG"])

        gs = GridSpec(4, 2, width_ratios=[2.2, 1],
                      left=0.05, right=0.97, top=0.87, bottom=0.07,
                      hspace=0.38, wspace=0.10, figure=fig)

        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        ax3d.set_facecolor(THEME["BG"])

        DG0_p = DG0_m[:, :tc]
        T_p   = T_m[:, :tc]
        K_p   = K_surf[:, :tc] * zs

        ax3d.plot_surface(
            DG0_p, T_p, K_p,
            cmap=CMAP_MARCUS, norm=norm,
            alpha=0.90, rstride=2, cstride=2,
            edgecolor=(0.0, 0.95, 1.0, 0.06), linewidth=0.18,
            antialiased=True, zorder=1,
        )

        z_floor = -0.04 * zs
        ax3d.contourf(DG0_p, T_p, K_p, zdir="z", offset=z_floor,
                      cmap=CMAP_MARCUS, norm=norm, alpha=0.25, levels=10)

        ax3d.plot(dG0_ridge[:tc], t_v[:tc], np.ones(tc) * zs,
                  color=THEME["ORANGE"], lw=2.8, alpha=0.95, zorder=15)
        ax3d.scatter([dG0_ridge[tc - 1]], [t_v[tc - 1]], [zs],
                     s=28, color=THEME["YELLOW"], edgecolor="white",
                     linewidth=0.5, zorder=20)

        ax3d.plot(dG0_v[:tc], t_v[:tc], k_on_surf[:tc] * zs,
                  color=THEME["CYAN"], lw=1.8, alpha=0.85, zorder=14)

        vis_inv = inv_m[:tc]
        if np.any(vis_inv):
            ax3d.scatter(
                dG0_v[:tc][vis_inv], t_v[:tc][vis_inv],
                k_on_surf[:tc][vis_inv] * zs + 0.03,
                s=35, color=THEME["YELLOW"], marker="*",
                zorder=18, edgecolor="white", linewidths=0.4,
            )

        pane_c = (0.02, 0.02, 0.02, 1.0)
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.set_pane_color(pane_c)
            axis._axinfo["grid"]["color"] = (0.13, 0.13, 0.13, 0.5)
            axis._axinfo["grid"]["linewidth"] = 0.35

        ax3d.set_xlabel(r"$\Delta G^0$", fontsize=9, fontweight="bold",
                        color=THEME["TEXT_DIM"], labelpad=10,
                        fontfamily=THEME["FONT"])
        ax3d.set_ylabel("TIME", fontsize=9, fontweight="bold",
                        color=THEME["TEXT_DIM"], labelpad=10,
                        fontfamily=THEME["FONT"])
        ax3d.set_zlabel(r"$\hat{k}$", fontsize=10, fontweight="bold",
                        color=THEME["TEXT_DIM"], labelpad=10,
                        fontfamily=THEME["FONT"])
        ax3d.set_zlim(-0.04, 1.10)
        ax3d.set_box_aspect([1.6, 2.0, 0.75])
        ax3d.view_init(elev=sched["elev"], azim=sched["azim"])
        ax3d.tick_params(axis="both", colors=THEME["TEXT_DIM"], labelsize=7)

        # ═══════════ INSET ═══════════
        ax_ins = fig.add_axes([0.055, 0.075, 0.20, 0.14], zorder=100)
        ax_ins.set_facecolor("#0a0a0a")
        for sp in ax_ins.spines.values():
            sp.set_color(THEME["SPINE"])
            sp.set_linewidth(0.5)
        ax_ins.tick_params(axis="both", colors=THEME["TEXT_DIM"],
                           labelsize=5, direction="in")

        ax_ins.imshow(K2D.T, origin="lower", aspect="auto",
                      cmap=CMAP_MARCUS, vmin=0, vmax=1,
                      extent=[dG0_2d[0], dG0_2d[-1],
                              lam_2d[0], lam_2d[-1]])

        diag_l = np.linspace(lam_2d[0], lam_2d[-1], 200)
        ax_ins.plot(-diag_l, diag_l,
                    color=THEME["ORANGE"], lw=1.0, ls="--", alpha=0.8)

        if tc > 1:
            pts = np.array([dG0_v[:tc], lam_v[:tc]]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap="cool", linewidth=0.9, alpha=0.85)
            lc.set_array(np.linspace(0, 1, tc - 1))
            ax_ins.add_collection(lc)

        ax_ins.set_xlabel(r"$\Delta G^0$", fontsize=5,
                          color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])
        ax_ins.set_ylabel(r"$\lambda$", fontsize=5,
                          color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])
        ax_ins.set_title("2D MARCUS MAP", fontsize=5,
                         color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])

        # ═══════════ RIGHT PANELS ═══════════
        panels = [fig.add_subplot(gs[i, 1]) for i in range(4)]
        for p in panels:
            _style_ax(p)

        ax1 = panels[0]
        ax1.plot(t_v[:tc], dG0_v[:tc], color=THEME["CYAN"], lw=0.9)
        ax1.plot(t_v[:tc], -lam_v[:tc], color=THEME["ORANGE"],
                 lw=0.9, ls="--")
        ax1.fill_between(t_v[:tc], dG0_v[:tc], -lam_v[:tc],
                         where=(dG0_v[:tc] < -lam_v[:tc]),
                         color=THEME["RED"], alpha=0.15)
        ax1.set_title("DRIVING FORCE  " + r"$\Delta G^0(t)$",
                      fontsize=8, color=THEME["TEXT_DIM"],
                      fontfamily=THEME["FONT"], pad=3)

        ax2 = panels[1]
        ax2.plot(t_v[:tc], lam_v[:tc], color=THEME["YELLOW"], lw=0.9,
                 label=r"$\lambda$")
        # kBT_v is already scaled by c_lam in main.py
        ax2.plot(t_v[:tc], kBT_v[:tc], color=THEME["GREEN"], lw=0.9,
                 alpha=0.7, label=r"$k_BT$")
        ax2.set_title("REORGANIZATION ENERGY  " + r"$\lambda(t)$",
                      fontsize=8, color=THEME["TEXT_DIM"],
                      fontfamily=THEME["FONT"], pad=3)
        leg2 = ax2.legend(loc="upper left", fontsize=6,
                          facecolor=THEME["BG"], edgecolor=THEME["GRID"])
        for txt in leg2.get_texts():
            txt.set_color(THEME["TEXT_DIM"])

        ax3 = panels[2]
        ax3.plot(t_v[:tc], dG_dd[:tc], color=THEME["MAGENTA"], lw=0.9)
        ax3.fill_between(t_v[:tc], 0, dG_dd[:tc],
                         color=THEME["MAGENTA"], alpha=0.20)
        ax3.axhline(0, color="#ffffff", lw=0.3, alpha=0.3)
        ax3.set_title("ACTIVATION BARRIER  " + r"$\Delta G^\ddagger(t)$",
                      fontsize=8, color=THEME["TEXT_DIM"],
                      fontfamily=THEME["FONT"], pad=3)

        ax4 = panels[3]
        ax4.plot(t_v[:tc], k_mar[:tc], color=THEME["ORANGE"], lw=0.9)
        ax4.plot(t_v[:tc], k_act[:tc], color=THEME["CYAN"], lw=0.9)
        ax4.set_ylim(-0.05, 1.10)
        ax4.set_title(r"$k_{Marcus}$ (ORANGE)  vs  $k_{actual}$ (CYAN)",
                      fontsize=8, color=THEME["TEXT_DIM"],
                      fontfamily=THEME["FONT"], pad=3)
        vc = np.isfinite(k_act[:tc]) & np.isfinite(k_mar[:tc])
        if np.sum(vc) > 10:
            cc = np.corrcoef(k_mar[:tc][vc], k_act[:tc][vc])[0, 1]
            ax4.text(0.97, 0.92, f"corr={cc:.2f}",
                     transform=ax4.transAxes, ha="right", va="top",
                     fontsize=7, color=THEME["YELLOW"],
                     fontfamily=THEME["FONT"], fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2",
                               facecolor="#0d0d0d",
                               edgecolor=THEME["GRID"], alpha=0.8))

        # ═══════════ TITLE BLOCK ═══════════
        n_inv_so_far = int(np.sum(inv_m[:tc]))

        fig.text(0.50, 0.960,
                 "MARCUS THEORY OF CAPITAL ROTATION  —  "
                 "INVERTED REGION DYNAMICS",
                 ha="center", va="center",
                 fontsize=24, fontweight="bold",
                 color=THEME["ORANGE"], fontfamily=THEME["FONT"])

        fig.text(0.50, 0.932,
                 f"$\\hat{{k}}(\\Delta G^0,t)=\\exp("
                 f"-(\\Delta G^0+\\lambda)^2/4\\lambda k_BT)$"
                 f"     $\\lambda_{{mean}}={lam_mean:.4f}$"
                 f"     $k_BT_{{mean}}={kBT_mn:.4f}$"
                 f"     INVERTED EVENTS: {n_inv_so_far:d}",
                 ha="center", va="center",
                 fontsize=10, color=THEME["TEXT_DIM"],
                 fontfamily=THEME["FONT"])

        idx_now = min(tc - 1, T_eff - 1)
        phase_str = _determine_phase(dG0_v[idx_now], lam_v[idx_now])

        fig.text(0.96, 0.900,
                 f"$\\Delta G^0$={dG0_v[idx_now]:.4f}    "
                 f"$\\lambda$={lam_v[idx_now]:.4f}    "
                 f"$\\Delta G^{{\\ddagger}}$={dG_dd[idx_now]:.5f}    "
                 f"PHASE: {phase_str}",
                 ha="right", va="center",
                 fontsize=10, fontweight="bold",
                 color=THEME["YELLOW"], fontfamily=THEME["FONT"])

        fig.text(0.985, 0.010, CONFIG["WATERMARK"],
                 ha="right", va="bottom", fontsize=10,
                 color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"],
                 alpha=0.6)

        # ═══════════ PROGRESS BAR ═══════════
        bar_y = 0.013
        bar_h = 0.004
        bar_x0, bar_x1 = 0.05, 0.97
        prog = fi / max(1, total_frames - 1)

        fig.patches.append(Rectangle(
            (bar_x0, bar_y), bar_x1 - bar_x0, bar_h,
            transform=fig.transFigure, facecolor="#1a1a1a",
            edgecolor="none", zorder=50))

        phase_name = sched["phase"]
        if phase_name == "GROW":
            bar_color = THEME["ORANGE"]
        elif phase_name == "HOLD":
            bar_color = THEME["YELLOW"]
        else:
            bar_color = THEME["CYAN"]

        fig.patches.append(Rectangle(
            (bar_x0, bar_y), (bar_x1 - bar_x0) * prog, bar_h,
            transform=fig.transFigure, facecolor=bar_color,
            edgecolor="none", alpha=0.7, zorder=51))

        frames.append(_canvas_to_rgb(fig))
        plt.close(fig)

        if (fi + 1) % 20 == 0 or fi == 0:
            print(f"    frame {fi + 1:3d}/{total_frames}  "
                  f"[{phase_name}]  tc={tc}  "
                  f"elev={sched['elev']:.1f}  azim={sched['azim']:.1f}")

    imageio.mimsave(CONFIG["OUT_GIF"], frames, fps=10, loop=0)
    print(f"  [ANIMATE] GIF saved -> {CONFIG['OUT_GIF']}")
    print(f"  [ANIMATE] {total_frames} frames, 10 FPS, "
          f"{total_frames / 10:.1f}s loop")