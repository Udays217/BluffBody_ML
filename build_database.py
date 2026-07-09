#!/usr/bin/env python3
"""
build_database.py
-----------------
Builds HDF5 snapshot databases for autoencoder training from Fluent ASCII exports.

Output per Reynolds number  →  database_Re{re}.h5  containing:
  coords            float32  [N, 2]          x, y of every filtered node
  data              float32  [N, 160, 2]     data[node, snapshot, var]
                                               var 0 = pressure
                                               var 1 = vorticity_magnitude
  snapshot_index    int32    [160]            0 … 159
  cycle             int32    [160]            1 … 10
  snapshot_in_cycle int32    [160]            1 … 16

Spatial filter: x ∈ [-3, 10],  y ∈ [-3, 3]
"""

# ── auto-install h5py into the active environment if missing ──────────────────
import subprocess, sys
try:
    import h5py
except ImportError:
    print("h5py not found — installing now …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py"])
    import h5py

import numpy as np
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(
    r"C:\Users\udays\OneDrive - Shiv Nadar Institution of Eminence"
    r"\2026 Summer Internships\IITM\10-cycle-validation-data"
)
OUT_DIR = Path(
    r"C:\Users\udays\OneDrive - Shiv Nadar Institution of Eminence"
    r"\2026 Summer Internships\IITM\ML model"
)

X_MIN, X_MAX        = -3.0, 10.0
Y_MIN, Y_MAX        = -3.0,  3.0

N_SNAPSHOTS         = 160
N_CYCLES            = 10
SNAPSHOTS_PER_CYCLE = 16

# folder, file prefix, first timestep number, step between files
RE_CONFIG = {
    120: dict(folder="10-cyc-120", prefix="FFF-1-", start=15084, step=36),
     90: dict(folder="10-cyc-90",  prefix="FFF-1-", start=18772, step=38),
     70: dict(folder="10-cyc-70",  prefix="FFF-1-", start=21930, step=43),
}

# Fluent export column indices (0-based):
#  0=nodenumber  1=x  2=y  3=pressure  4=vel-mag  5=x-vel  6=y-vel  7=vorticity-mag
COL_X    = 1
COL_Y    = 2
COL_P    = 3
COL_VORT = 7

# ── Helpers ───────────────────────────────────────────────────────────────────
def file_list(folder: Path, prefix: str, start: int, step: int, n: int = 160):
    """Return ordered list of snapshot Paths."""
    return [folder / f"{prefix}{start + i * step}" for i in range(n)]


def read_xyp_vort(path: Path) -> np.ndarray:
    """Read x, y, pressure, vorticity from one file → float32 array [n_nodes, 4]."""
    return np.genfromtxt(
        path, delimiter=",", skip_header=1,
        usecols=(COL_X, COL_Y, COL_P, COL_VORT),
        dtype=np.float32,
    )


def read_pv_only(path: Path) -> np.ndarray:
    """Read pressure and vorticity only → float32 array [n_nodes, 2]."""
    return np.genfromtxt(
        path, delimiter=",", skip_header=1,
        usecols=(COL_P, COL_VORT),
        dtype=np.float32,
    )


# ── Main builder ──────────────────────────────────────────────────────────────
def build(re: int, cfg: dict):
    folder = BASE_DIR / cfg["folder"]
    files  = file_list(folder, cfg["prefix"], cfg["start"], cfg["step"])

    bar = "=" * 56
    print(f"\n{bar}")
    print(f"  Reynolds number : {re}")
    print(f"  Source folder   : {folder.name}")
    print(f"  Files           : {len(files)}")
    print(bar)

    # ── Step 1: spatial mask (built once from snapshot 0) ──────────────────
    print("  [1/4]  Building spatial filter from snapshot 0 ...", end=" ", flush=True)
    arr0 = read_xyp_vort(files[0])

    mask = (
        (arr0[:, 0] >= X_MIN) & (arr0[:, 0] <= X_MAX) &
        (arr0[:, 1] >= Y_MIN) & (arr0[:, 1] <= Y_MAX)
    )
    n_nodes = int(mask.sum())
    print(f"done  →  {n_nodes} / {len(arr0)} nodes kept")

    coords = arr0[mask, :2].copy()   # [n_nodes, 2]

    # ── Step 2: allocate output tensor ─────────────────────────────────────
    tensor = np.empty((n_nodes, N_SNAPSHOTS, 2), dtype=np.float32)

    # Fill snapshot 0 from data already in memory
    tensor[:, 0, 0] = arr0[mask, 2]   # pressure
    tensor[:, 0, 1] = arr0[mask, 3]   # vorticity

    # ── Step 3: read remaining 159 snapshots ───────────────────────────────
    print(f"  [2/4]  Reading {N_SNAPSHOTS} snapshots ...")
    for j, fpath in enumerate(files[1:], start=1):
        pv = read_pv_only(fpath)
        tensor[:, j, 0] = pv[mask, 0]
        tensor[:, j, 1] = pv[mask, 1]
        if j % SNAPSHOTS_PER_CYCLE == 0:
            print(f"         snapshot {j:3d}/{N_SNAPSHOTS}  "
                  f"(end of cycle {j // SNAPSHOTS_PER_CYCLE})")

    # ── Step 4: metadata arrays ────────────────────────────────────────────
    print("  [3/4]  Assembling metadata ...", end=" ", flush=True)
    snap_idx       = np.arange(N_SNAPSHOTS, dtype=np.int32)
    cycle          = np.repeat(
                        np.arange(1, N_CYCLES + 1),
                        SNAPSHOTS_PER_CYCLE
                     ).astype(np.int32)
    snap_in_cycle  = np.tile(
                        np.arange(1, SNAPSHOTS_PER_CYCLE + 1),
                        N_CYCLES
                     ).astype(np.int32)
    print("done")

    # ── Step 5: write HDF5 ─────────────────────────────────────────────────
    out_path = OUT_DIR / f"database_Re{re}.h5"
    print(f"  [4/4]  Saving → {out_path.name} ...", end=" ", flush=True)

    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("coords",
                          data=coords,
                          compression="gzip", compression_opts=4)
        hf.create_dataset("data",
                          data=tensor,
                          compression="gzip", compression_opts=4)
        hf.create_dataset("snapshot_index",    data=snap_idx)
        hf.create_dataset("cycle",             data=cycle)
        hf.create_dataset("snapshot_in_cycle", data=snap_in_cycle)

        # Attributes — useful metadata for later ML code
        hf.attrs["reynolds_number"]      = re
        hf.attrs["n_nodes"]              = n_nodes
        hf.attrs["n_snapshots"]          = N_SNAPSHOTS
        hf.attrs["n_cycles"]             = N_CYCLES
        hf.attrs["snapshots_per_cycle"]  = SNAPSHOTS_PER_CYCLE
        hf.attrs["x_range"]              = [X_MIN, X_MAX]
        hf.attrs["y_range"]              = [Y_MIN, Y_MAX]
        hf.attrs["variables"]            = ["pressure", "vorticity_magnitude"]
        hf.attrs["data_shape"]           = "data[node_idx, snapshot_idx, var_idx]"
        hf.attrs["var_0"]                = "pressure"
        hf.attrs["var_1"]                = "vorticity_magnitude"

    mb = out_path.stat().st_size / 1e6
    print(f"done  ({mb:.1f} MB)")
    print(f"         coords : {coords.shape}  |  data : {tensor.shape}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for re, cfg in RE_CONFIG.items():
        build(re, cfg)

    print(f"\n{'=' * 56}")
    print("  All 3 databases built successfully.\n")
    for re in RE_CONFIG:
        p = OUT_DIR / f"database_Re{re}.h5"
        print(f"    database_Re{re}.h5   ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"\n  Output directory: {OUT_DIR}\n")
