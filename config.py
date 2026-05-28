"""
Marcus Theory of Capital Rotation — Configuration Module
Theme, Marcus parameters, custom colormap, and global CONFIG dict.
"""

from matplotlib.colors import LinearSegmentedColormap

# ════════════════════════════════════════════════════════════════
# THEME — Bloomberg Dark Standard
# ════════════════════════════════════════════════════════════════
THEME = {
    "BG":           "#000000",
    "PANEL_BG":     "#0a0a0a",
    "GRID":         "#1a1a1a",
    "SPINE":        "#333333",
    "TEXT":         "#ffffff",
    "TEXT_DIM":     "#aaaaaa",
    "ORANGE":       "#ff9500",
    "ORANGE_HOT":   "#ff6b00",
    "CYAN":         "#00f2ff",
    "YELLOW":       "#ffd400",
    "GREEN":        "#00ff41",
    "RED":          "#ff3050",
    "MAGENTA":      "#ff1493",
    "PINK":         "#ff2a9e",
    "BLUE":         "#00bfff",
    "FONT":         "Arial",
}

# ════════════════════════════════════════════════════════════════
# CUSTOM COLORMAP — Marcus Rate Surface
# Inverted side (left of ridge): deep violet → purple → magenta
# Normal side (right of ridge): orange → yellow → white-hot
# ════════════════════════════════════════════════════════════════
CMAP_MARCUS = LinearSegmentedColormap.from_list("marcus_rate", [
    "#0a0010",   # Near-black with deep violet tint — deep inverted region
    "#4a0080",   # Purple — inverted region
    "#ff1493",   # Magenta — approaching activationless from inverted side
    "#ff9500",   # Orange — approaching activationless from normal side
    "#ffd400",   # Yellow — near activationless
    "#ffffff",   # White-hot — AT activationless (maximum rate)
])

# ════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# ════════════════════════════════════════════════════════════════
CONFIG = {
    "N_STOCKS":         30,
    "N_SECTORS":        6,
    "STOCKS_PER_SECTOR": 5,
    "T_TOTAL":          756,
    "ROLL_MOMENTUM":    252,
    "ROLL_LAMBDA":      21,
    "ROLL_KBT":         21,
    "ROLL_ROTATION":    5,
    "N_TOP_BOT":        2,
    "N_DG0_GRID":       100,
    "N_LAM_GRID":       60,
    "C_DELTA_G":        1.0,
    "C_LAMBDA":         0.8,
    "DPI_PNG":          100,
    "DPI_GIF":          80,
    "FIG_W":            19.2,
    "FIG_H":            10.8,
    "OUT_DIR":          "outputs",
    "OUT_PNG":          "outputs/marcus_capital_rotation.png",
    "OUT_GIF":          "outputs/marcus_capital_rotation.gif",
    "WATERMARK":        "@Laksh",
    "FONT":             "Arial",
    "SEED":             42,
    "RHO_WITHIN":       0.45,
    "RHO_ACROSS":       0.20,
    "SIGMA_BASE":       0.20,
    "GARCH_OMEGA":      1e-5,
    "GARCH_ALPHA":      0.10,
    "GARCH_BETA":       0.85,
}

SECTOR_NAMES = [
    "Technology", "Financials", "Healthcare",
    "Energy", "Consumer Disc.", "Industrials",
]

if __name__ == "__main__":
    print("CONFIG loaded successfully.")
    print(f"  THEME keys: {len(THEME)}")
    print(f"  CONFIG keys: {len(CONFIG)}")
    print(f"  CMAP_MARCUS: {CMAP_MARCUS.name}, {len(CMAP_MARCUS.colors)} stops")