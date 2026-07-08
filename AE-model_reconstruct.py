import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.patches as mpatches

BASE = Path(__file__).resolve().parent

DB_FILES = {
    100: BASE / "database_Re100.h5",
    80: BASE / "database_Re80.h5",
    60: BASE / "database_Re60.h5",
}

MODEL_SAVE_PATH = BASE / "AE_self.pt"
LOSS_PLOT_PATH = BASE / "loss_curves.png"
CONTOUR_DIR = BASE / "contour_comparisons"

LATENT_DIM = 32
HIDDEN_1 = 512
HIDDEN_2 = 128
EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_all_data():
    all_snapshots = []
    re_labels = []
    cycle_labels = []
    coords = None
    for Re, path in DB_FILES.items():
        with h5py.File(path, "r") as hf:
            data = hf["data"][:]
            cycles = hf["cycle"][:]
            if coords is None:
                coords = hf["coords"][:]
            n_nodes, n_snaps, n_vars = data.shape
            for j in range(n_snaps):
                snapshot = data[:, j, :]
                flat = snapshot.reshape(-1)
                all_snapshots.append(flat)
                re_labels.append(Re)
                cycle_labels.append(cycles[j])
    all_data = np.stack(all_snapshots, axis=0)
    re_labels = np.array(re_labels)
    cycles = np.array(cycle_labels)
    return coords, all_data, re_labels, cycles, n_nodes

def split_by_cycle(all_data , re_labels, cycles):
    train_mask = cycles <= 8
    val_mask = cycles == 9
    test_mask = cycles == 10
    return {
        "train": (all_data[train_mask], re_labels[train_mask]),
        "val": (all_data[val_mask], re_labels[val_mask]),
        "test": (all_data[test_mask], re_labels[test_mask]),
    }

class MinMaxScaler:
    def __init__(self):
        self.data_min = None
        self.data_max = None
        self.data_range = None
    
    def fit(self, X: np.ndarray):
        self.data_min = X.min(axis=0)
        self.data_max = X.max(axis=0)
        self.data_range = self.data_max - self.data_min
        self.data_range[self.data_range == 0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.data_min) / self.data_range

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.data_range + self.data_min
    
class FlowDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]
    
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_2, LATENT_DIM),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_2, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
def train_model(model, train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            x_hat = model(batch)
            loss = criterion(x_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train = epoch_loss / n_batches
        train_losses.append(avg_train)
        model.eval()

        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                x_hat = model(batch)
                loss = criterion(x_hat, batch)
                val_loss += loss.item()
                n_val += 1
        avg_val = val_loss / n_val
        val_losses.append(avg_val)

        if epoch % 15 == 0 or epoch == 1:
            print(f" Epoch {epoch:3d}/{EPOCHS} "
                  f"train_loss={avg_train:.6f} val_loss={avg_val:.6f}")
    return train_losses, val_losses
    
def plot_loss_curves(train_losses, val_losses):
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

def make_triangulation(coords):
    x, y = coords[:, 0], coords[:, 1]
    triang = mtri.Triangulation(x, y)
    xm = x[triang.triangles].mean(axis=1)
    ym = y[triang.triangles].mean(axis=1)
    return triang.set_mask(xm**2 + ym**2 < 0.5**2)

def add_cylinder(ax):
    ax.add_patch(mpatches.Circle((0, 0), 0.5, color="#444444", zorder=10))

def _contour_ax(ax, triang, values, var, title):
    if var == "pressure":
        lim = max(abs(float(values.min())), abs(float(values.max())))
        cf = ax.tricontourf(triang, values, levels=64, cmap="RdBu_r", 
                            vmin=-lim, vmax=lim)
        ax.tricontour(triang, values, levels=12, 
                      colors="k", linewidths=0.3, alpha=0.35)
    else:
        vmax = float(np.percentile(values, 98))
        cf = ax.tricontourf(triang, values, levels=64, cmap="inferno", 
                            vmin=0.0, vmax=max(vmax, 1e-8))
    
    plt.colorbar(cf, ax=ax, shrink=0.85)
    add_cylinder(ax)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-3, 10)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("x / D", fontsize=7)
    ax.set_ylabel("y / D", fontsize=7)

def plot_comparison_contours(model, scaler, coords, all_data, re_labels,
                             snapshot_indices, n_nodes):
    CONTOUR_DIR.mkdir(exist_ok=True)
    triang = make_triangulation(coords)
    model.eval()
    for snap_idx in snapshot_indices:
        fig, axes = plt.subplots(3, 4, figsize=(20, 11))
        fig.suptitle(f"Original vs Reconstructed  —  snapshot index {snap_idx}",
                     fontsize=14, fontweight="bold")

        for row, re in enumerate([100, 80, 60]):
            re_positions = np.where(re_labels == re)[0]
            global_idx   = re_positions[snap_idx]

            orig_flat = all_data[global_idx]                    
            orig_2d   = orig_flat.reshape(n_nodes, 2)           
            orig_p    = orig_2d[:, 0]                          
            orig_v    = orig_2d[:, 1]                    

            scaled    = scaler.transform(orig_flat[np.newaxis, :])  
            inp       = torch.from_numpy(scaled).float().to(DEVICE)
            with torch.no_grad():
                recon_scaled = model(inp).cpu().numpy()             
            recon_flat = scaler.inverse_transform(recon_scaled)[0]  
            recon_2d   = recon_flat.reshape(n_nodes, 2)
            recon_p    = recon_2d[:, 0]
            recon_v    = recon_2d[:, 1]

            _contour_ax(axes[row, 0], triang, orig_p,  "pressure",
                        f"Re={re} | Original Pressure")
            _contour_ax(axes[row, 1], triang, recon_p, "pressure",
                        f"Re={re} | Reconstructed Pressure")
            _contour_ax(axes[row, 2], triang, orig_v,  "vorticity",
                        f"Re={re} | Original Vorticity")
            _contour_ax(axes[row, 3], triang, recon_v, "vorticity",
                        f"Re={re} | Reconstructed Vorticity")

        plt.tight_layout()
        out_path = CONTOUR_DIR / f"comparison_idx{snap_idx}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Comparison contour saved → {out_path.relative_to(BASE)}")

def save_model(model, scaler, n_nodes):
    torch,np.save({
        "model_state_dict": model.state_dict(),
        "scaler_min": scaler.data_min,
        "scaler_max": scaler.data_max,
        "scaler_range": scaler.data_range,
        "n_nodes": n_nodes,
        "input_dim": n_nodes * 2,
        "latent_dim": LATENT_DIM
    }, MODEL_SAVE_PATH)

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

def reconstruct_and_export(model, scaler, n_nodes, re_list, snap_idx, var):
    """
    Run a snapshot through the autoencoder and save:
      1. an HDF5 file with original + reconstructed data
      2. a comparison contour plot
    """
    CONTOUR_DIR.mkdir(exist_ok=True)
    triang = None
    coords = None

    for re in re_list:
        with h5py.File(DB_FILES[re], "r") as hf:
            orig_2d = hf["data"][:, snap_idx, :]      # [N, 2]
            if coords is None:
                coords = hf["coords"][:]
                triang = make_triangulation(coords)

        orig_flat = orig_2d.reshape(-1)                 # [N*2]

        scaled = scaler.transform(orig_flat[np.newaxis, :])
        inp    = torch.from_numpy(scaled).float().to(DEVICE)
        with torch.no_grad():
            recon_scaled = model(inp).cpu().numpy()
        recon_flat = scaler.inverse_transform(recon_scaled)[0]
        recon_2d   = recon_flat.reshape(n_nodes, 2)

        h5_path = CONTOUR_DIR / f"reconstructed_Re{re}_idx{snap_idx}.h5"
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("coords",       data=coords)
            hf.create_dataset("original",     data=orig_2d)
            hf.create_dataset("reconstructed", data=recon_2d)
            hf.attrs["reynolds_number"] = re
            hf.attrs["snapshot_index"]  = snap_idx
            hf.attrs["var_0"]           = "pressure"
            hf.attrs["var_1"]           = "vorticity_magnitude"
        print(f"  HDF5 saved → {h5_path.relative_to(BASE)}")

        vars_to_plot = (["pressure", "vorticity"] if var == "both"
                        else [var])
        n_vars = len(vars_to_plot)
        fig, axes = plt.subplots(n_vars, 2, figsize=(14, 5 * n_vars))
        axes = np.atleast_2d(axes)

        fig.suptitle(f"Re = {re}  |  snapshot index {snap_idx}",
                     fontsize=13, fontweight="bold")

        for v_row, v_name in enumerate(vars_to_plot):
            v_idx = 0 if v_name == "pressure" else 1
            _contour_ax(axes[v_row, 0], triang, orig_2d[:, v_idx],
                        v_name, f"Original {v_name.capitalize()}")
            _contour_ax(axes[v_row, 1], triang, recon_2d[:, v_idx],
                        v_name, f"Reconstructed {v_name.capitalize()}")

        plt.tight_layout()
        png_path = CONTOUR_DIR / f"reconstructed_Re{re}_idx{snap_idx}_{var}.png"
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Contour saved → {png_path.relative_to(BASE)}")


def cycle_snap_to_index(re, cycle_num, snap_num):
    """Convert (cycle, snapshot_in_cycle) to flat index 0–159."""
    with h5py.File(DB_FILES[re], "r") as hf:
        c   = hf["cycle"][:]
        sic = hf["snapshot_in_cycle"][:]
    hits = np.where((c == cycle_num) & (sic == snap_num))[0]
    if len(hits) == 0:
        raise ValueError(f"No snapshot for cycle={cycle_num}, snap={snap_num}")
    return int(hits[0])

def main():
    parser = argparse.ArgumentParser(description="Flow field autoencoder")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "reconstruct"],
                        help="'train' or 'reconstruct'")
    parser.add_argument("--re",    type=str, default="all",
                        help="60, 80, 100, or 'all'")
    parser.add_argument("--var",   type=str, default="both",
                        choices=["pressure", "vorticity", "both"])
    parser.add_argument("--index", type=int, default=None,
                        help="Flat snapshot index 0–159")
    parser.add_argument("--cycle", type=int, default=None)
    parser.add_argument("--snap",  type=int, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        print(f"\n  Device: {DEVICE}")
        print(f"  Loading data ...")

        coords, all_data, re_labels, cycles, n_nodes = load_all_data()
        input_dim = n_nodes * 2         
        print(f"  Samples: {all_data.shape[0]}   Features: {input_dim}")

        splits = split_by_cycle(all_data, re_labels, cycles)
        train_data, train_re = splits["train"]
        val_data,   val_re   = splits["val"]
        test_data,  test_re  = splits["test"]
        print(f"  Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")

        scaler = MinMaxScaler()
        scaler.fit(train_data)

        train_scaled = scaler.transform(train_data)
        val_scaled   = scaler.transform(val_data)

        train_loader = DataLoader(FlowDataset(train_scaled),
                                  batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(FlowDataset(val_scaled),
                                  batch_size=BATCH_SIZE, shuffle=False)

        model = Autoencoder(input_dim).to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {total_params:,}")
        print(f"  Training for {EPOCHS} epochs ...\n")

        train_losses, val_losses = train_model(model, train_loader, val_loader)

        model.eval()
        test_scaled = scaler.transform(test_data)
        test_tensor = torch.from_numpy(test_scaled).float().to(DEVICE)
        with torch.no_grad():
            recon = model(test_tensor)
            test_loss = nn.MSELoss()(recon, test_tensor).item()
        print(f"\n  Test loss (cycle 10): {test_loss:.6f}")

        save_model(model, scaler, n_nodes)

        plot_loss_curves(train_losses, val_losses)

        print(f"\n  Generating comparison contours ...")
        plot_comparison_contours(model, scaler, coords, all_data,
                                 re_labels, [0, 50, 100, 159], n_nodes)

        print(f"\n  Training complete.\n")

    elif args.mode == "reconstruct":
        model, scaler, n_nodes = load_model()

        re_list = ([100, 80, 60] if args.re == "all"
                   else [int(args.re)])

        if args.index is not None:
            snap_idx = args.index
        elif args.cycle is not None and args.snap is not None:
            snap_idx = cycle_snap_to_index(re_list[0], args.cycle, args.snap)
        else:
            parser.error("Provide --index or both --cycle and --snap")

        print(f"\n  Reconstructing: Re={args.re}  index={snap_idx}  var={args.var}")
        reconstruct_and_export(model, scaler, n_nodes, re_list, snap_idx, args.var)
        print(f"\n  Done.\n")


if __name__ == "__main__":
    main()