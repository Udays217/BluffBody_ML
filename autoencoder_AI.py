"""
autoencoder.py
--------------
Fully-connected autoencoder for 2D cylinder wake flow fields.

Modes
-----
  Train (also saves the latent code of every snapshot to latents.h5):
    python autoencoder.py --mode train

  Save latents only (rebuild latents.h5 from a trained model, no retraining):
    python autoencoder.py --mode save-latents

  Generate  (PURE DECODE — reads the stored latent code, never the original field):
    python autoencoder.py --mode generate --re 100 --index 47 --var both
    python autoencoder.py --mode generate --re all --cycle 3 --snap 8 --var pressure

  NOTE: generate mode outputs the GENERATED field only. To view the original
  for visual comparison, run plot_contours.py separately for the same snapshot.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS AND CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

import argparse
import subprocess
import sys
from pathlib import Path

# auto-install matplotlib if missing (h5py, numpy, torch already present)
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend for saving figures
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib
    matplotlib.use("Agg")

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent             # ML model folder

DB_FILES = {
    100: BASE / "database_Re100.h5",
     80: BASE / "database_Re80.h5",
     60: BASE / "database_Re60.h5",
}

MODEL_SAVE_PATH  = BASE / "autoencoder_model.pt"   # trained weights + scaler
LATENTS_PATH     = BASE / "latents.h5"             # stored latent codes (the "z database")
LOSS_PLOT_PATH   = BASE / "loss_curves.png"
OUTPUT_DIR       = BASE / "generated"              # generated fields + contours land here

# ── Hyperparameters ───────────────────────────────────────────────────────────
LATENT_DIM    = 32
HIDDEN_1      = 512
HIDDEN_2      = 128
EPOCHS        = 300
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-5                               # L2 regularisation

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_data():
    """
    Read all 3 HDF5 databases.

    Returns
    -------
    coords    : np.ndarray [N, 2]         — shared node coordinates
    all_data  : np.ndarray [480, N*2]     — every snapshot flattened
    re_labels : np.ndarray [480]          — which Re each sample belongs to
    cycles    : np.ndarray [480]          — cycle number (1–10) for each sample
    n_nodes   : int                       — number of spatial nodes (N)
    """
    all_snapshots = []      # will hold flattened snapshot vectors
    re_labels     = []      # Re tag for each snapshot
    cycle_labels  = []      # cycle tag for each snapshot
    coords        = None    # set once (same mesh for all Re)

    for re, path in DB_FILES.items():
        with h5py.File(path, "r") as hf:
            data   = hf["data"][:]          # [N, 160, 2]
            cycle  = hf["cycle"][:]         # [160]

            if coords is None:
                coords = hf["coords"][:]    # [N, 2]

            n_nodes, n_snaps, n_vars = data.shape       # 15543, 160, 2

            for j in range(n_snaps):
                # flatten node × variable into one long vector:
                # [p0, v0, p1, v1, ..., p_N, v_N]  length = N * 2
                snapshot = data[:, j, :]                # [N, 2]
                flat     = snapshot.reshape(-1)         # [N*2]
                all_snapshots.append(flat)
                re_labels.append(re)
                cycle_labels.append(cycle[j])

    all_data  = np.stack(all_snapshots, axis=0)         # [480, N*2]
    re_labels = np.array(re_labels)                     # [480]
    cycles    = np.array(cycle_labels)                   # [480]

    return coords, all_data, re_labels, cycles, n_nodes


def split_by_cycle(all_data, re_labels, cycles):
    """
    Split into train (cycles 1-8), val (cycle 9), test (cycle 10).
    Returns data arrays and their corresponding Re labels.
    """
    train_mask = cycles <= 8
    val_mask   = cycles == 9
    test_mask  = cycles == 10

    return {
        "train": (all_data[train_mask], re_labels[train_mask]),
        "val":   (all_data[val_mask],   re_labels[val_mask]),
        "test":  (all_data[test_mask],  re_labels[test_mask]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MIN-MAX SCALER
# ═══════════════════════════════════════════════════════════════════════════════

class MinMaxScaler:
    """
    Scales each feature to [0, 1] based on training-set min/max.

    Stores data_min and data_max as 1-D arrays of length N*2
    so that pressure columns and vorticity columns each get their
    own scaling range.
    """

    def __init__(self):
        self.data_min = None      # [N*2]
        self.data_max = None      # [N*2]
        self.data_range = None    # max - min, with zeros replaced by 1

    def fit(self, X: np.ndarray):
        """Compute min and max from training data X of shape [n_samples, N*2]."""
        self.data_min   = X.min(axis=0)                       # min across samples
        self.data_max   = X.max(axis=0)                       # max across samples
        self.data_range = self.data_max - self.data_min
        # avoid division by zero for nodes where the value never changes
        self.data_range[self.data_range == 0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale X to [0, 1]."""
        return (X - self.data_min) / self.data_range

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Undo scaling back to physical values."""
        return X * self.data_range + self.data_min


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PYTORCH DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class FlowDataset(Dataset):
    """
    Wraps a numpy array of shape [n_samples, N*2] as a PyTorch Dataset.
    Each __getitem__ returns a float32 tensor — the input IS the target
    for an autoencoder, so we only store one copy.
    """

    def __init__(self, data: np.ndarray):
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — AUTOENCODER MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class Autoencoder(nn.Module):
    """
    Symmetric fully-connected autoencoder.

    Encoder: input_dim → 512 → 128 → 32  (latent)
    Decoder:       32  → 128 → 512 → input_dim

    ReLU between hidden layers.
    No activation on the final output (continuous regression).
    """

    def __init__(self, input_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_2, LATENT_DIM),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_2, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, input_dim),
            # no activation — raw output for regression
        )

    def forward(self, x):
        z = self.encoder(x)       # compress to latent vector
        x_hat = self.decoder(z)   # reconstruct from latent
        return x_hat

    def encode(self, x):
        """Return just the latent representation."""
        return self.encoder(x)

    def decode(self, z):
        """Reconstruct from a latent vector."""
        return self.decoder(z)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader):
    """
    Train the autoencoder.

    Returns lists of per-epoch training and validation losses.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_losses = []
    val_losses   = []

    for epoch in range(1, EPOCHS + 1):

        # ── Training phase ─────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)

            x_hat = model(batch)            # forward pass
            loss  = criterion(x_hat, batch) # MSE between input and reconstruction

            optimizer.zero_grad()           # clear old gradients
            loss.backward()                 # backpropagation
            optimizer.step()                # update weights

            epoch_loss += loss.item()
            n_batches  += 1

        avg_train = epoch_loss / n_batches
        train_losses.append(avg_train)

        # ── Validation phase ───────────────────────────────────────────────
        model.eval()
        val_loss   = 0.0
        n_val      = 0

        with torch.no_grad():               # no gradient computation needed
            for batch in val_loader:
                batch = batch.to(DEVICE)
                x_hat = model(batch)
                loss  = criterion(x_hat, batch)
                val_loss += loss.item()
                n_val    += 1

        avg_val = val_loss / n_val
        val_losses.append(avg_val)

        # ── Progress printing ──────────────────────────────────────────────
        if epoch % 25 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss={avg_train:.6f}  val_loss={avg_val:.6f}")

    return train_losses, val_losses


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — LOSS CURVE PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_loss_curves(train_losses, val_losses):
    """Save training vs validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_losses, label="Train", linewidth=1.2)
    ax.plot(val_losses,   label="Validation", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training & Validation Loss")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close()
    print(f"  Loss curves saved → {LOSS_PLOT_PATH.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — CONTOUR PLOTTING (single field — generated only)
# ═══════════════════════════════════════════════════════════════════════════════

def make_triangulation(coords):
    """Delaunay triangulation with cylinder interior masked."""
    x, y   = coords[:, 0], coords[:, 1]
    triang = mtri.Triangulation(x, y)
    xm = x[triang.triangles].mean(axis=1)
    ym = y[triang.triangles].mean(axis=1)
    triang.set_mask(xm**2 + ym**2 < 0.5**2)
    return triang


def add_cylinder(ax):
    ax.add_patch(mpatches.Circle((0, 0), 0.5, color="#444444", zorder=10))


def _contour_ax(ax, triang, values, var, title):
    """Draw one contour subplot."""
    if var == "pressure":
        lim = max(abs(float(values.min())), abs(float(values.max())))
        cf  = ax.tricontourf(triang, values, levels=64, cmap="RdBu_r",
                             vmin=-lim, vmax=lim)
        ax.tricontour(triang, values, levels=12,
                      colors="k", linewidths=0.3, alpha=0.35)
    else:
        vmax = float(np.percentile(values, 98))
        cf   = ax.tricontourf(triang, values, levels=64, cmap="inferno",
                              vmin=0, vmax=max(vmax, 1e-8))
    plt.colorbar(cf, ax=ax, shrink=0.85)
    add_cylinder(ax)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-3, 10)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("x / D", fontsize=7)
    ax.set_ylabel("y / D", fontsize=7)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SAVE AND LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def save_model(model, scaler, n_nodes):
    """Save model weights and scaler parameters to a single .pt file."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_min":       scaler.data_min,
        "scaler_max":       scaler.data_max,
        "scaler_range":     scaler.data_range,
        "n_nodes":          n_nodes,
        "input_dim":        n_nodes * 2,
        "latent_dim":       LATENT_DIM,
    }, MODEL_SAVE_PATH)
    print(f"  Model saved → {MODEL_SAVE_PATH.name}")


def load_model():
    """Load trained model and scaler from disk."""
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)

    input_dim = checkpoint["input_dim"]
    n_nodes   = checkpoint["n_nodes"]

    model = Autoencoder(input_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = MinMaxScaler()
    scaler.data_min   = checkpoint["scaler_min"]
    scaler.data_max   = checkpoint["scaler_max"]
    scaler.data_range = checkpoint["scaler_range"]

    return model, scaler, n_nodes


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — SAVE LATENT CODES  (build the "z database")
# ═══════════════════════════════════════════════════════════════════════════════

def save_latents(model, scaler, coords, all_data, re_labels, cycles, n_nodes):
    """
    Encode every snapshot and store its latent vector to latents.h5.

    This is what makes pure generation possible: after this runs, the decoder
    can rebuild any snapshot from its stored 32-dim code WITHOUT ever touching
    the original field again.

    latents.h5 layout
    ------------------
      z          float32  [480, LATENT_DIM]   one latent code per snapshot
      re         int32    [480]               Reynolds number of each code
      snap_index int32    [480]               snapshot index 0–159 of each code
      coords     float32  [N, 2]              node coordinates (for later plotting)
    """
    model.eval()

    # scale the full dataset, push through the ENCODER only
    scaled  = scaler.transform(all_data)                        # [480, N*2]
    tensor  = torch.from_numpy(scaled).float().to(DEVICE)
    with torch.no_grad():
        z = model.encode(tensor).cpu().numpy()                  # [480, LATENT_DIM]

    # snapshot index within each Re block: 0..159 repeated per Re.
    # all_data is laid out Re100(0..159), Re80(0..159), Re60(0..159),
    # so np.where per Re preserves that 0..159 order.
    snap_index = np.zeros(len(re_labels), dtype=np.int32)
    for re in np.unique(re_labels):
        positions = np.where(re_labels == re)[0]
        snap_index[positions] = np.arange(len(positions))

    with h5py.File(LATENTS_PATH, "w") as hf:
        hf.create_dataset("z",          data=z.astype(np.float32))
        hf.create_dataset("re",         data=re_labels.astype(np.int32))
        hf.create_dataset("snap_index", data=snap_index)
        hf.create_dataset("coords",     data=coords.astype(np.float32))
        hf.attrs["latent_dim"] = LATENT_DIM
        hf.attrs["n_nodes"]    = n_nodes
        hf.attrs["var_0"]      = "pressure"
        hf.attrs["var_1"]      = "vorticity_magnitude"

    print(f"  Latent codes saved → {LATENTS_PATH.name}   (z shape {z.shape})")


def lookup_latent(re, snap_idx):
    """
    Fetch the stored latent vector for one (Re, snapshot index) from latents.h5.

    Returns
    -------
    z_vec  : np.ndarray [LATENT_DIM]   the stored latent code
    coords : np.ndarray [N, 2]         node coordinates
    """
    with h5py.File(LATENTS_PATH, "r") as hf:
        z_all      = hf["z"][:]              # [480, LATENT_DIM]
        re_all     = hf["re"][:]             # [480]
        snap_all   = hf["snap_index"][:]     # [480]
        coords     = hf["coords"][:]         # [N, 2]

    hits = np.where((re_all == re) & (snap_all == snap_idx))[0]
    if len(hits) == 0:
        raise ValueError(f"No stored latent for Re={re}, snapshot index={snap_idx}")
    return z_all[hits[0]], coords


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — GENERATE FROM LATENT  (pure decode, no original field)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_from_latent(model, scaler, n_nodes, re_list, snap_idx, var):
    """
    PURE GENERATION.

    For each requested Re:
      1. look up the stored latent code z for (Re, snap_idx)   ← no original field read
      2. decode z → scaled field, then un-scale to physical values
      3. save the generated field as HDF5 and a contour PNG

    The original snapshot is never loaded. Use plot_contours.py if you want to
    eyeball the original alongside these generated outputs.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    model.eval()

    triang = None
    coords = None

    for re in re_list:
        # ── 1. look up the latent code (this is the ONLY data we read) ──────
        z_vec, coords_from_file = lookup_latent(re, snap_idx)   # [LATENT_DIM], [N,2]
        if coords is None:
            coords = coords_from_file
            triang = make_triangulation(coords)

        # ── 2. decode z → physical field ───────────────────────────────────
        z_tensor = torch.from_numpy(z_vec[np.newaxis, :]).float().to(DEVICE)  # [1, LATENT_DIM]
        with torch.no_grad():
            gen_scaled = model.decode(z_tensor).cpu().numpy()   # [1, N*2]
        gen_flat = scaler.inverse_transform(gen_scaled)[0]      # [N*2]
        gen_2d   = gen_flat.reshape(n_nodes, 2)                 # [N, 2]

        # ── 3a. save generated field as HDF5 ───────────────────────────────
        h5_path = OUTPUT_DIR / f"generated_Re{re}_idx{snap_idx}.h5"
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("coords",    data=coords)
            hf.create_dataset("generated", data=gen_2d)     # [N, 2] pressure|vorticity
            hf.create_dataset("latent_z",  data=z_vec)      # the code it came from
            hf.attrs["reynolds_number"] = re
            hf.attrs["snapshot_index"]  = snap_idx
            hf.attrs["var_0"]           = "pressure"
            hf.attrs["var_1"]           = "vorticity_magnitude"
            hf.attrs["note"]            = "pure decode output — no original field"
        print(f"  HDF5 saved → {h5_path.relative_to(BASE)}")

        # ── 3b. save contour(s) of the generated field ─────────────────────
        vars_to_plot = (["pressure", "vorticity"] if var == "both" else [var])
        n_vars = len(vars_to_plot)
        fig, axes = plt.subplots(1, n_vars, figsize=(7 * n_vars, 4.5))
        axes = np.atleast_1d(axes)

        fig.suptitle(f"Generated (decoded)  |  Re = {re}  |  snapshot index {snap_idx}",
                     fontsize=12, fontweight="bold")

        for col, v_name in enumerate(vars_to_plot):
            v_idx = 0 if v_name == "pressure" else 1
            _contour_ax(axes[col], triang, gen_2d[:, v_idx],
                        v_name, f"Generated {v_name.capitalize()}")

        plt.tight_layout()
        png_path = OUTPUT_DIR / f"generated_Re{re}_idx{snap_idx}_{var}.png"
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Contour saved → {png_path.relative_to(BASE)}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — CYCLE/SNAP → INDEX HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def cycle_snap_to_index(re, cycle_num, snap_num):
    """Convert (cycle, snapshot_in_cycle) to flat index 0–159."""
    with h5py.File(DB_FILES[re], "r") as hf:
        c   = hf["cycle"][:]
        sic = hf["snapshot_in_cycle"][:]
    hits = np.where((c == cycle_num) & (sic == snap_num))[0]
    if len(hits) == 0:
        raise ValueError(f"No snapshot for cycle={cycle_num}, snap={snap_num}")
    return int(hits[0])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Flow field autoencoder")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "save-latents", "generate"],
                        help="'train', 'save-latents', or 'generate'")
    parser.add_argument("--re",    type=str, default="all",
                        help="60, 80, 100, or 'all'")
    parser.add_argument("--var",   type=str, default="both",
                        choices=["pressure", "vorticity", "both"])
    parser.add_argument("--index", type=int, default=None,
                        help="Flat snapshot index 0–159")
    parser.add_argument("--cycle", type=int, default=None)
    parser.add_argument("--snap",  type=int, default=None)
    args = parser.parse_args()

    # ══════════════════════════════════════════════════════════════════════
    #  TRAIN MODE
    # ══════════════════════════════════════════════════════════════════════
    if args.mode == "train":
        print(f"\n  Device: {DEVICE}")
        print(f"  Loading data ...")

        coords, all_data, re_labels, cycles, n_nodes = load_all_data()
        input_dim = n_nodes * 2         # 15543 * 2 = 31086
        print(f"  Samples: {all_data.shape[0]}   Features: {input_dim}")

        # ── split ──────────────────────────────────────────────────────────
        splits = split_by_cycle(all_data, re_labels, cycles)
        train_data, train_re = splits["train"]
        val_data,   val_re   = splits["val"]
        test_data,  test_re  = splits["test"]
        print(f"  Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")

        # ── scale ──────────────────────────────────────────────────────────
        scaler = MinMaxScaler()
        scaler.fit(train_data)

        train_scaled = scaler.transform(train_data)
        val_scaled   = scaler.transform(val_data)

        # ── dataloaders ────────────────────────────────────────────────────
        train_loader = DataLoader(FlowDataset(train_scaled),
                                  batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(FlowDataset(val_scaled),
                                  batch_size=BATCH_SIZE, shuffle=False)

        # ── build and train ────────────────────────────────────────────────
        model = Autoencoder(input_dim).to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {total_params:,}")
        print(f"  Training for {EPOCHS} epochs ...\n")

        train_losses, val_losses = train_model(model, train_loader, val_loader)

        # ── test set evaluation ────────────────────────────────────────────
        model.eval()
        test_scaled = scaler.transform(test_data)
        test_tensor = torch.from_numpy(test_scaled).float().to(DEVICE)
        with torch.no_grad():
            recon = model(test_tensor)
            test_loss = nn.MSELoss()(recon, test_tensor).item()
        print(f"\n  Test loss (cycle 10): {test_loss:.6f}")

        # ── save model ─────────────────────────────────────────────────────
        save_model(model, scaler, n_nodes)

        # ── save the latent codes for every snapshot (the z database) ──────
        save_latents(model, scaler, coords, all_data, re_labels, cycles, n_nodes)

        # ── loss curves ────────────────────────────────────────────────────
        plot_loss_curves(train_losses, val_losses)

        print(f"\n  Training complete.\n")

    # ══════════════════════════════════════════════════════════════════════
    #  SAVE-LATENTS MODE  (rebuild latents.h5 without retraining)
    # ══════════════════════════════════════════════════════════════════════
    elif args.mode == "save-latents":
        model, scaler, n_nodes = load_model()
        coords, all_data, re_labels, cycles, n_nodes = load_all_data()
        save_latents(model, scaler, coords, all_data, re_labels, cycles, n_nodes)
        print(f"\n  Done.\n")

    # ══════════════════════════════════════════════════════════════════════
    #  GENERATE MODE  (pure decode from stored latent codes)
    # ══════════════════════════════════════════════════════════════════════
    elif args.mode == "generate":
        model, scaler, n_nodes = load_model()

        re_list = ([100, 80, 60] if args.re == "all"
                   else [int(args.re)])

        # resolve snapshot index
        if args.index is not None:
            snap_idx = args.index
        elif args.cycle is not None and args.snap is not None:
            snap_idx = cycle_snap_to_index(re_list[0], args.cycle, args.snap)
        else:
            parser.error("Provide --index or both --cycle and --snap")

        print(f"\n  Generating (pure decode): Re={args.re}  index={snap_idx}  var={args.var}")
        generate_from_latent(model, scaler, n_nodes, re_list, snap_idx, args.var)
        print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
