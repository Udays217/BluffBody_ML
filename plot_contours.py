"""
plot_contours.py
----------------
Generate pressure / vorticity contour plots from the HDF5 databases.

Usage examples
--------------
# Single variable — Re=100, cycle 3, snapshot 7, pressure:
python plot_contours.py --re 100 --cycle 3 --snap 7 --var pressure

# Vorticity instead:
python plot_contours.py --re 100 --cycle 1 --snap 1 --var vorticity

# Both variables side by side (3 Re x 2 vars — same layout as the overview image):
python plot_contours.py --re all --cycle 1 --snap 1 --var both

# Compare all three Re for one variable:
python plot_contours.py --re all --cycle 5 --snap 8 --var vorticity

# Use a flat snapshot index (0-159) instead of cycle+snap:
python plot_contours.py --re 80 --index 32 --var vorticity

# Save to a specific filename:
python plot_contours.py --re 100 --cycle 2 --snap 4 --var pressure --out my_plot.png
"""

import argparse
import subprocess, sys

try:
    import matplotlib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.patches as mpatches
import h5py
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent

DB = {
    100: BASE / "database_Re100.h5",
     80: BASE / "database_Re80.h5",
     60: BASE / "database_Re60.h5",
}

CMAPS  = {"pressure": "RdBu_r", "vorticity": "inferno"}
LABELS = {"pressure": "p",      "vorticity": "|ω|"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_snapshot(re: int, snap_idx: int):
    """Return coords [N,2], pressure [N], vorticity [N] for one snapshot."""
    with h5py.File(DB[re], "r") as hf:
        coords = hf["coords"][:]
        pres   = hf["data"][:, snap_idx, 0]
        vort   = hf["data"][:, snap_idx, 1]
    return coords, pres, vort


def cycle_snap_to_index(re: int, cycle: int, snap_in_cycle: int) -> int:
    """Convert (cycle, snap_in_cycle) to flat snapshot index."""
    with h5py.File(DB[re], "r") as hf:
        c   = hf["cycle"][:]
        sic = hf["snapshot_in_cycle"][:]
    hits = np.where((c == cycle) & (sic == snap_in_cycle))[0]
    if len(hits) == 0:
        raise ValueError(f"No snapshot found for cycle={cycle}, snap_in_cycle={snap_in_cycle}")
    return int(hits[0])


def make_triangulation(coords: np.ndarray):
    """Delaunay triangulation with cylinder interior masked out."""
    x, y   = coords[:, 0], coords[:, 1]
    triang = mtri.Triangulation(x, y)
    xm = x[triang.triangles].mean(axis=1)
    ym = y[triang.triangles].mean(axis=1)
    triang.set_mask(xm**2 + ym**2 < 0.5**2)   # cylinder radius = 0.5D
    return triang


def add_cylinder(ax):
    ax.add_patch(mpatches.Circle((0, 0), 0.5, color="#444444", zorder=10))


def plot_single(ax, triang, values, var: str, title: str):
    cmap = CMAPS[var]
    if var == "pressure":
        lim = max(abs(float(values.min())), abs(float(values.max())))
        cf  = ax.tricontourf(triang, values, levels=64, cmap=cmap,
                             vmin=-lim, vmax=lim)
        ax.tricontour(triang, values, levels=12,
                      colors="k", linewidths=0.3, alpha=0.35)
    else:
        vmax = float(np.percentile(values, 98))   # clip near-wall spikes
        cf   = ax.tricontourf(triang, values, levels=64, cmap=cmap,
                              vmin=0, vmax=vmax)

    plt.colorbar(cf, ax=ax, shrink=0.85, label=LABELS[var])
    add_cylinder(ax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x / D")
    ax.set_ylabel("y / D")
    ax.set_aspect("equal")
    ax.set_xlim(-3, 10)
    ax.set_ylim(-3, 3)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot pressure / vorticity contours")
    parser.add_argument("--re",    type=str, default="100",
                        help="Reynolds number: 60, 80, 100, or 'all'")
    parser.add_argument("--var",   type=str, default="pressure",
                        choices=["pressure", "vorticity", "both"],
                        help="Variable to plot. 'both' gives a side-by-side overview")
    parser.add_argument("--index", type=int, default=None,
                        help="Flat snapshot index 0-159 (overrides --cycle/--snap)")
    parser.add_argument("--cycle", type=int, default=1,
                        help="Cycle number 1-10")
    parser.add_argument("--snap",  type=int, default=1,
                        help="Snapshot within cycle 1-16")
    parser.add_argument("--out",   type=str, default=None,
                        help="Output filename (default: auto-generated)")
    args = parser.parse_args()

    re_list = [100, 80, 60] if args.re == "all" else [int(args.re)]

    # ── Resolve snapshot index ─────────────────────────────────────────────
    if args.index is not None:
        snap_indices = {re: args.index for re in re_list}
        label = f"snapshot index {args.index}"
    else:
        snap_indices = {re: cycle_snap_to_index(re, args.cycle, args.snap)
                        for re in re_list}
        label = f"cycle {args.cycle} · snapshot {args.snap}"

    n = len(re_list)

    # ── Build figure ───────────────────────────────────────────────────────
    if args.var == "both":
        # n rows × 2 columns — pressure left, vorticity right
        # this replicates the overview image shown in the conversation
        fig, axes = plt.subplots(n, 2, figsize=(14, 4.5 * n))
        axes = np.atleast_2d(axes)
        fig.suptitle(f"Pressure & Vorticity  —  {label}",
                     fontsize=13, fontweight="bold", y=1.01)
        for row, re in enumerate(re_list):
            coords, pres, vort = load_snapshot(re, snap_indices[re])
            triang = make_triangulation(coords)
            plot_single(axes[row, 0], triang, pres, "pressure",
                        title=f"Re = {re}  |  Pressure")
            plot_single(axes[row, 1], triang, vort, "vorticity",
                        title=f"Re = {re}  |  Vorticity")
    else:
        # n rows × 1 column — single variable
        fig, axes = (plt.subplots(n, 1, figsize=(12, 4.5 * n)) if n > 1
                     else plt.subplots(1, 1, figsize=(12, 4.5)))
        axes = np.atleast_1d(axes)
        fig.suptitle(f"{args.var.capitalize()}  —  {label}",
                     fontsize=13, fontweight="bold", y=1.01)
        for ax, re in zip(axes, re_list):
            coords, pres, vort = load_snapshot(re, snap_indices[re])
            triang = make_triangulation(coords)
            values = pres if args.var == "pressure" else vort
            plot_single(ax, triang, values, args.var,
                        title=f"Re = {re}  |  {args.var.capitalize()}")

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────
    if args.out:
        out = Path(args.out)
    else:
        re_tag   = args.re
        snap_tag = (f"idx{args.index}" if args.index is not None
                    else f"c{args.cycle}s{args.snap}")
        out = BASE / f"contour_Re{re_tag}_{args.var}_{snap_tag}.png"

    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
