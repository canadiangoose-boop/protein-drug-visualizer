"""
nn_scorer.py — Neural Network Binding Energy Scorer
=====================================================
Defines the MLP architecture and an inference wrapper (NNScorer) that
plugs directly into the visualizer as a drop-in replacement for the
classical PhysicsEngine.

Architecture
    Input  : pairwise distances between every drug atom and every protein atom
             10 drug × 13 protein = 130 features
    Network: 130 → 128 → 64 → 32 → 1
    Output : predicted ΔG in kcal/mol

The NNScorer.score() method returns the same dict shape as
PhysicsEngine.score(), so the visualizer needs zero changes to swap scorers.
Interaction lines still come from the physics engine (which runs in
parallel) so dashed bonds are always visible regardless of scoring mode.
"""

import os
import numpy as np
import torch
import torch.nn as nn

# ── Feature dimensions (must match molecular data in protein_visualizer.py) ──
N_DRUG_ATOMS    = 10
N_PROTEIN_ATOMS = 13
N_FEATURES      = N_DRUG_ATOMS * N_PROTEIN_ATOMS   # 130


# ═══════════════════════════════════════════════════════════════════════════
# Model definition
# ═══════════════════════════════════════════════════════════════════════════

class BindingEnergyMLP(nn.Module):
    """
    Multi-layer perceptron predicting molecular binding energy.

    Layers
    ------
    Linear(130→128) → ReLU → BatchNorm → Dropout(0.1)
    Linear(128→ 64) → ReLU → BatchNorm → Dropout(0.1)
    Linear( 64→ 32) → ReLU → BatchNorm → Dropout(0.1)
    Linear( 32→  1)                                     → ΔG (kcal/mol)

    BatchNorm stabilises training on the wide energy range.
    Dropout prevents overfitting given the relatively simple dataset.
    """

    def __init__(self, input_dim: int = N_FEATURES,
                 hidden_dims: tuple = (128, 64, 32),
                 dropout: float = 0.1):
        super().__init__()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# Inference wrapper
# ═══════════════════════════════════════════════════════════════════════════

class NNScorer:
    """
    Drop-in replacement for PhysicsEngine — loads a trained model and
    predicts binding energy from the current drug/protein configuration.

    Usage
    -----
        scorer = NNScorer()                        # loads models/binding_scorer.pt
        energy = scorer.predict(drug_atoms, protein_atoms)
        result = scorer.score(drug_atoms, protein_atoms)  # returns full dict
    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'binding_scorer.pt')
    STATS_PATH = os.path.join(os.path.dirname(__file__), 'models', 'binding_scorer_stats.pt')

    def __init__(self, model_path: str = None, stats_path: str = None):
        mp = model_path or self.MODEL_PATH
        sp = stats_path or self.STATS_PATH

        self.device = torch.device('cpu')
        self.model  = BindingEnergyMLP()
        self.model.load_state_dict(torch.load(mp, map_location=self.device))
        self.model.eval()

        stats = torch.load(sp, map_location=self.device)
        self.feat_mean  = stats['feat_mean'].numpy()
        self.feat_std   = stats['feat_std'].numpy()
        self.label_mean = float(stats['label_mean'])
        self.label_std  = float(stats['label_std'])

        print(f"[NNScorer] Loaded model from {mp}")
        print(f"[NNScorer] Label stats: mean={self.label_mean:.2f}  std={self.label_std:.2f}")

    # ── Feature extraction ────────────────────────────────────────────────

    @staticmethod
    def extract_features(drug_atoms: list, protein_atoms: list) -> np.ndarray:
        """
        Build the 130-d feature vector: flattened pairwise distance matrix.

        Rows = drug atoms (10), Columns = protein atoms (13).
        Features are simply Euclidean distances in Ångstroms.
        The network learns which distances correspond to favourable contacts.
        """
        feat = []
        for da in drug_atoms:
            for pa in protein_atoms:
                feat.append(float(np.linalg.norm(da.position - pa.position)))
        return np.array(feat, dtype=np.float32)

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, drug_atoms: list, protein_atoms: list) -> float:
        """Returns predicted ΔG in kcal/mol."""
        feat = self.extract_features(drug_atoms, protein_atoms)
        feat_norm = (feat - self.feat_mean) / (self.feat_std + 1e-8)
        x = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_norm = self.model(x).item()
        return pred_norm * self.label_std + self.label_mean

    def score(self, drug_atoms: list, protein_atoms: list) -> dict:
        """
        Returns a result dict matching PhysicsEngine.score() shape.
        The visualizer uses this for the energy HUD.
        Interaction lines are drawn separately via the physics engine,
        so they remain visible in NN mode.
        """
        total = self.predict(drug_atoms, protein_atoms)
        return {
            'vdw': 0.0, 'electrostatic': 0.0,
            'hbonds': 0.0, 'hydrophobic': 0.0,
            'total': total,
            'n_hbonds': 0,
            'interactions': [],  # NN gives no per-pair breakdown
            '_mode': 'nn',
        }

    # ── Availability check ────────────────────────────────────────────────

    @classmethod
    def is_available(cls) -> bool:
        return os.path.exists(cls.MODEL_PATH) and os.path.exists(cls.STATS_PATH)
