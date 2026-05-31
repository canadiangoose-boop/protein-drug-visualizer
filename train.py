#!/usr/bin/env python3
"""
train.py — Train the ML Binding Energy Scorer
===============================================
Generates a synthetic dataset using the physics engine, trains an MLP,
and logs every experiment to Weights & Biases.

Run
---
    python train.py                    # default hyperparameters
    python train.py --epochs 200       # custom epochs
    python train.py --sweep            # W&B hyperparameter sweep

What gets logged to W&B
-----------------------
  • Training + validation loss curves (per epoch)
  • MAE and R² on the validation set
  • Actual vs predicted scatter plot
  • Trained model saved as a W&B Artifact
  • Full hyperparameter config

W&B dashboard
-------------
  After running, visit: https://wandb.ai/<your-username>/protein-drug-binding
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import wandb

# ── Local imports ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from protein_visualizer import (
    PhysicsEngine, make_protein_atoms, make_drug_atoms, APPROACH_DISTANCE
)
from nn_scorer import BindingEnergyMLP, NNScorer, N_FEATURES

os.makedirs('models', exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# §1  DEFAULT HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # ── Dataset ─────────────────────────────────────────────────
    "n_samples":    6000,    # number of random drug poses to generate
    "val_split":    0.20,    # fraction held out for validation
    "energy_clip":  40.0,    # kcal/mol — clips extreme LJ repulsion spikes
    "approach_min": 0.15,    # min approach_t (don't sample too-far poses)

    # ── Architecture ────────────────────────────────────────────
    "hidden_dims":  [128, 64, 32],
    "dropout":      0.10,
    "activation":   "ReLU",

    # ── Training ─────────────────────────────────────────────────
    "n_epochs":     150,
    "batch_size":   128,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "optimizer":    "AdamW",
    "scheduler":    "CosineAnnealingLR",

    # ── W&B metadata ─────────────────────────────────────────────
    "feature_type": "pairwise_distances_130d",
    "scorer_type":  "MLP",
    "data_source":  "physics_engine_synthetic",
}


# ═══════════════════════════════════════════════════════════════════════════
# §2  DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def place_drug(drug_atoms: list, approach_t: float, rotate_deg: float):
    """Apply approach + rotation transform to drug atoms (mirrors visualizer logic)."""
    rad     = np.radians(rotate_deg)
    cos_r, sin_r = np.cos(rad), np.sin(rad)
    Ry = np.array([[cos_r, 0, sin_r],
                   [0,     1, 0    ],
                   [-sin_r,0, cos_r]])

    centroid     = np.mean([a._origin for a in drug_atoms], axis=0)
    approach_vec = np.array([0.0, -APPROACH_DISTANCE, 0.0])

    for atom in drug_atoms:
        rel          = atom._origin - centroid
        rotated      = Ry @ rel + centroid
        atom.position = rotated + approach_t * approach_vec


def generate_dataset(n_samples: int,
                     approach_min: float,
                     energy_clip: float) -> tuple:
    """
    Generate (features, labels) by sampling random drug poses.

    Sampling strategy
    -----------------
    approach_t  ~ Uniform[approach_min, 1.0]  (drug moves toward pocket)
    rotate_deg  ~ Uniform[0, 360]             (full rotational space)

    Features: 130-d pairwise distance matrix (drug × protein atoms)
    Labels:   total binding energy from PhysicsEngine, clipped to ±energy_clip

    Returns (X, y) as float32 numpy arrays.
    """
    print(f"\n[Data] Generating {n_samples} samples ...")
    t0 = time.time()

    physics = PhysicsEngine()
    protein = make_protein_atoms()

    features, labels = [], []

    for i in range(n_samples):
        drug = make_drug_atoms()

        approach_t = np.random.uniform(approach_min, 1.0)
        rotate_deg = np.random.uniform(0.0, 360.0)
        place_drug(drug, approach_t, rotate_deg)

        scores = physics.score(drug, protein)
        energy = float(np.clip(scores['total'], -energy_clip, energy_clip))

        feat = NNScorer.extract_features(drug, protein)
        features.append(feat)
        labels.append(energy)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_samples}  ({elapsed:.1f}s)")

    X = np.array(features, dtype=np.float32)
    y = np.array(labels,   dtype=np.float32)

    print(f"[Data] Done in {time.time()-t0:.1f}s")
    print(f"[Data] Energy range: {y.min():.2f} → {y.max():.2f} kcal/mol")
    print(f"[Data] Energy mean:  {y.mean():.2f} ± {y.std():.2f}")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# §3  NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════

def normalise(X_train, X_val, y_train, y_val):
    """Z-score features and labels using training-set statistics only."""
    feat_mean = X_train.mean(axis=0)
    feat_std  = X_train.std(axis=0) + 1e-8

    label_mean = float(y_train.mean())
    label_std  = float(y_train.std()) + 1e-8

    X_tr_n = (X_train - feat_mean) / feat_std
    X_vl_n = (X_val   - feat_mean) / feat_std
    y_tr_n = (y_train  - label_mean) / label_std
    y_vl_n = (y_val    - label_mean) / label_std

    stats = {
        'feat_mean':  torch.tensor(feat_mean,  dtype=torch.float32),
        'feat_std':   torch.tensor(feat_std,   dtype=torch.float32),
        'label_mean': torch.tensor(label_mean, dtype=torch.float32),
        'label_std':  torch.tensor(label_std,  dtype=torch.float32),
    }

    return X_tr_n, X_vl_n, y_tr_n, y_vl_n, stats


# ═══════════════════════════════════════════════════════════════════════════
# §4  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train(config: dict = None):
    """
    Full training pipeline:
      1. Init W&B run
      2. Generate dataset
      3. Train MLP
      4. Log metrics, scatter plot, model artifact
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # ── W&B init ────────────────────────────────────────────────────────
    run = wandb.init(
        project = "protein-drug-binding",
        config  = cfg,
        tags    = ["mlp", "binding-energy", "synthetic-data"],
        notes   = "MLP trained on physics-engine synthetic data. "
                  "Input: pairwise drug–protein distances. Output: ΔG kcal/mol.",
    )
    cfg = wandb.config   # allows W&B sweep to override values

    # ── Dataset ─────────────────────────────────────────────────────────
    X, y = generate_dataset(cfg.n_samples, cfg.approach_min, cfg.energy_clip)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.val_split, random_state=42)

    X_tr_n, X_vl_n, y_tr_n, y_vl_n, stats = normalise(
        X_train, X_val, y_train, y_val)

    wandb.log({
        "data/n_train":      len(X_train),
        "data/n_val":        len(X_val),
        "data/energy_mean":  float(y_train.mean()),
        "data/energy_std":   float(y_train.std()),
        "data/energy_min":   float(y_train.min()),
        "data/energy_max":   float(y_train.max()),
    })

    # ── DataLoaders ──────────────────────────────────────────────────────
    def to_loader(X_np, y_np, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X_np, dtype=torch.float32),
            torch.tensor(y_np, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle)

    train_loader = to_loader(X_tr_n, y_tr_n, shuffle=True)
    val_loader   = to_loader(X_vl_n, y_vl_n, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────────
    model     = BindingEnergyMLP(
        input_dim   = N_FEATURES,
        hidden_dims = tuple(cfg.hidden_dims),
        dropout     = cfg.dropout,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.learning_rate,
        weight_decay = cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs, eta_min=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] Parameters: {n_params:,}")
    wandb.log({"model/n_params": n_params})

    # ── Training loop ────────────────────────────────────────────────────
    best_val_loss = float('inf')
    best_model_path = 'models/binding_scorer_best.pt'

    print(f"\n[Train] Starting {cfg.n_epochs} epochs ...\n")

    for epoch in range(1, cfg.n_epochs + 1):

        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_tr_n)

        # Validate
        model.eval()
        val_loss  = 0.0
        preds_all = []
        actuals_all = []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
                preds_all.extend(pred.numpy().tolist())
                actuals_all.extend(yb.numpy().tolist())
        val_loss /= len(X_vl_n)

        # Denormalise for human-readable metrics
        label_std  = float(stats['label_std'])
        label_mean = float(stats['label_mean'])
        preds_kcal   = np.array(preds_all)   * label_std + label_mean
        actuals_kcal = np.array(actuals_all) * label_std + label_mean

        val_mae = float(np.mean(np.abs(preds_kcal - actuals_kcal)))
        val_r2  = float(r2_score(actuals_kcal, preds_kcal))

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── W&B logging ─────────────────────────────────────────────────
        wandb.log({
            "train/loss":  train_loss,
            "val/loss":    val_loss,
            "val/mae_kcal": val_mae,
            "val/r2":      val_r2,
            "lr":          current_lr,
            "epoch":       epoch,
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{cfg.n_epochs} | "
                  f"train={train_loss:.4f}  val={val_loss:.4f} | "
                  f"MAE={val_mae:.2f} kcal/mol  R²={val_r2:.3f} | "
                  f"lr={current_lr:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    # ── Final model + stats ───────────────────────────────────────────────
    model_path = 'models/binding_scorer.pt'
    stats_path = 'models/binding_scorer_stats.pt'

    # Load best checkpoint
    model.load_state_dict(torch.load(best_model_path))
    torch.save(model.state_dict(), model_path)
    torch.save(stats, stats_path)

    print(f"\n[Train] Best val loss: {best_val_loss:.4f}")
    print(f"[Train] Model saved to: {model_path}")

    # ── Final validation metrics ─────────────────────────────────────────
    model.eval()
    final_preds, final_actuals = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            p = model(xb).numpy()
            final_preds.extend(p.tolist())
            final_actuals.extend(yb.numpy().tolist())

    fp_kcal = np.array(final_preds)   * label_std + label_mean
    fa_kcal = np.array(final_actuals) * label_std + label_mean
    final_mae = float(np.mean(np.abs(fp_kcal - fa_kcal)))
    final_r2  = float(r2_score(fa_kcal, fp_kcal))

    wandb.log({
        "final/val_mae_kcal": final_mae,
        "final/val_r2":       final_r2,
        "final/val_loss":     best_val_loss,
    })
    print(f"[Final] MAE = {final_mae:.3f} kcal/mol  |  R² = {final_r2:.4f}")

    # ── Scatter plot: actual vs predicted ────────────────────────────────
    scatter_table = wandb.Table(
        columns=["Actual ΔG (kcal/mol)", "Predicted ΔG (kcal/mol)"],
        data=[[float(a), float(p)] for a, p in zip(fa_kcal, fp_kcal)],
    )
    wandb.log({
        "actual_vs_predicted": wandb.plot.scatter(
            scatter_table,
            "Actual ΔG (kcal/mol)",
            "Predicted ΔG (kcal/mol)",
            title="Binding Energy: Actual vs Predicted",
        )
    })

    # ── Sample predictions table ─────────────────────────────────────────
    sample_table = wandb.Table(
        columns=["Actual (kcal/mol)", "Predicted (kcal/mol)", "Error (kcal/mol)"],
        data=[
            [round(float(a), 2), round(float(p), 2), round(float(p-a), 2)]
            for a, p in zip(fa_kcal[:50], fp_kcal[:50])
        ],
    )
    wandb.log({"sample_predictions": sample_table})

    # ── W&B Artifact: save model ─────────────────────────────────────────
    artifact = wandb.Artifact(
        name        = "binding-energy-scorer",
        type        = "model",
        description = (
            f"MLP binding energy scorer. "
            f"Val MAE={final_mae:.3f} kcal/mol, R²={final_r2:.4f}. "
            f"Trained on {cfg.n_samples} physics-engine samples."
        ),
        metadata = {
            "val_mae_kcal": final_mae,
            "val_r2":       final_r2,
            "n_samples":    int(cfg.n_samples),
            "n_params":     n_params,
            "hidden_dims":  list(cfg.hidden_dims),
        }
    )
    artifact.add_file(model_path)
    artifact.add_file(stats_path)
    run.log_artifact(artifact)
    print(f"[W&B] Model artifact logged.")

    entity = getattr(wandb.run, 'entity', None) or '<your-username>'
    wandb.finish()
    print("\n✓ Training complete.")
    print(f"  View results at: https://wandb.ai/{entity}/protein-drug-binding\n")
    print("  To sync offline run to cloud:  wandb sync wandb/offline-run-*/\n")

    return model_path


# ═══════════════════════════════════════════════════════════════════════════
# §5  HYPERPARAMETER SWEEP CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

SWEEP_CONFIG = {
    "method": "bayes",   # Bayesian optimisation — smarter than random search
    "metric": {"name": "val/mae_kcal", "goal": "minimize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-4, "max": 1e-2,
        },
        "hidden_dims": {
            "values": [
                [64, 32],
                [128, 64, 32],
                [256, 128, 64],
                [256, 128, 64, 32],
            ]
        },
        "dropout": {
            "values": [0.0, 0.05, 0.10, 0.20]
        },
        "batch_size": {
            "values": [64, 128, 256]
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-5, "max": 1e-2,
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# §6  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Train the ML binding energy scorer for the protein-drug visualizer."
    )
    p.add_argument('--epochs',    type=int,   default=DEFAULT_CONFIG['n_epochs'],
                   help='Number of training epochs')
    p.add_argument('--samples',   type=int,   default=DEFAULT_CONFIG['n_samples'],
                   help='Number of synthetic training samples to generate')
    p.add_argument('--lr',        type=float, default=DEFAULT_CONFIG['learning_rate'],
                   help='Learning rate')
    p.add_argument('--sweep',     action='store_true',
                   help='Run a W&B hyperparameter sweep instead of a single run')
    p.add_argument('--sweep-runs',type=int,   default=20,
                   help='Number of sweep runs (only used with --sweep)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.sweep:
        print("[Sweep] Starting W&B hyperparameter sweep ...")
        sweep_id = wandb.sweep(SWEEP_CONFIG, project="protein-drug-binding")
        wandb.agent(sweep_id, function=train, count=args.sweep_runs)

    else:
        overrides = {
            "n_epochs":     args.epochs,
            "n_samples":    args.samples,
            "learning_rate": args.lr,
        }
        train(config=overrides)
