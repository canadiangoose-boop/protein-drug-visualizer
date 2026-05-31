#!/usr/bin/env python3
"""
weave_eval.py — W&B Weave Evaluation for the Binding Energy Scorer
====================================================================
Uses W&B Weave to systematically evaluate and trace the NNScorer against
the classical PhysicsEngine across a structured test dataset of drug poses.

What this does
--------------
  1. Generates 72 diverse test poses (6 approach distances × 12 rotations)
  2. Computes physics-engine ground-truth energies for each pose
  3. Wraps the NN scorer and a random baseline as  weave.Model  objects
  4. Defines three @weave.op scorers:
       mae_scorer          — Mean Absolute Error vs physics (kcal/mol)
       direction_scorer    — Does NN agree on favorable vs unfavorable?
       induced_fit_scorer  — Does NN correctly predict threshold crossing?
  5. Runs  weave.Evaluation  on both models and logs everything to Weave

Weave dashboard
---------------
  After running (with W&B login), visit:
    https://wandb.ai/<username>/protein-drug-binding
  Click the "Weave" tab to see:
    • Per-example traces (inputs → output → scores)
    • Aggregate metrics table
    • Side-by-side NN vs Random baseline comparison

Run (online — requires W&B login)
---
    wandb login
    python weave_eval.py

Run (offline — full metrics, no Weave tracing)
---
    python weave_eval.py --local
"""

import os
import sys
import asyncio
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from protein_visualizer import (
    PhysicsEngine, make_protein_atoms, make_drug_atoms,
    APPROACH_DISTANCE, INDUCED_FIT_THRESHOLD,
)
from nn_scorer import NNScorer
from train import place_drug   # reuse the same pose-placement logic


# ═══════════════════════════════════════════════════════════════════════════
# §0  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weave evaluation for binding energy scorer")
    p.add_argument(
        "--local", action="store_true",
        help="Run evaluation locally without W&B Weave (no login required)"
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# §1  SHARED SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════

# Created once and reused across all model calls
_physics = PhysicsEngine()
_protein = make_protein_atoms()
_scorer  = NNScorer() if NNScorer.is_available() else None


# ═══════════════════════════════════════════════════════════════════════════
# §2  TEST DATASET
# ═══════════════════════════════════════════════════════════════════════════

def build_test_dataset() -> list[dict]:
    """
    Generates 72 test poses covering the full docking trajectory:

    approach_t  ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}   (far → docked)
    rotate_deg  ∈ {0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330}

    Physics-engine energies are pre-computed and stored as ground truth.
    Labels:
      label_favorable    — physics_energy < 0 (drug gains something by binding)
      label_induced_fit  — physics_energy < INDUCED_FIT_THRESHOLD (-7.5 kcal/mol)
    """
    print("[Dataset] Building 72-pose test dataset ...")
    rows = []
    approach_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rotation_values = list(range(0, 360, 30))   # every 30°

    for approach_t in approach_values:
        for rotate_deg in rotation_values:
            drug = make_drug_atoms()
            place_drug(drug, approach_t, rotate_deg)

            result = _physics.score(drug, _protein)
            energy = float(np.clip(result['total'], -50.0, 50.0))

            rows.append({
                "approach_t":        approach_t,
                "rotate_deg":        rotate_deg,
                "physics_energy":    energy,
                "n_hbonds":          result['n_hbonds'],
                "label_favorable":   energy < 0.0,
                "label_induced_fit": energy < INDUCED_FIT_THRESHOLD,
            })

    n_fav = sum(1 for r in rows if r['label_favorable'])
    n_if  = sum(1 for r in rows if r['label_induced_fit'])
    print(f"[Dataset] {len(rows)} poses | "
          f"{n_fav} favorable ({100*n_fav//len(rows)}%) | "
          f"{n_if} induced-fit ({100*n_if//len(rows)}%)")
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# §3  LOCAL (no-Weave) EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def _predict_local(approach_t: float, rotate_deg: float, mode: str) -> dict:
    """Run one prediction for the given mode ('nn', 'physics', 'random')."""
    drug = make_drug_atoms()
    place_drug(drug, approach_t, rotate_deg)

    if mode == "nn":
        if _scorer is None:
            raise RuntimeError("No trained model found. Run python train.py first.")
        energy = float(_scorer.predict(drug, _protein))
    elif mode == "physics":
        result = _physics.score(drug, _protein)
        energy = float(np.clip(result['total'], -50.0, 50.0))
    elif mode == "random":
        rng    = np.random.default_rng(42 + int(approach_t * 1000) + int(rotate_deg))
        energy = float(rng.uniform(-20.0, 50.0))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "energy":      round(energy, 3),
        "favorable":   bool(energy < 0.0),
        "induced_fit": bool(energy < INDUCED_FIT_THRESHOLD),
        "scorer":      mode,
    }


def run_local_evaluation(dataset: list[dict], mode: str, label: str) -> dict:
    """
    Pure-Python evaluation (no Weave).  Computes all four metrics locally and
    prints a summary identical to the Weave version.
    """
    print(f"\n[Eval] Running local evaluation for: {label}")

    mae_vals, dir_vals, if_vals, bias_vals = [], [], [], []

    for row in dataset:
        output = _predict_local(row["approach_t"], row["rotate_deg"], mode)

        mae_vals.append(abs(output["energy"] - row["physics_energy"]))
        dir_vals.append(int(output["favorable"]   == row["label_favorable"]))
        if_vals.append(int(output["induced_fit"]  == row["label_induced_fit"]))
        bias_vals.append(output["energy"] - row["physics_energy"])

    results = {
        "mae_scorer":          {"mae_kcal_mol":        {"mean": float(np.mean(mae_vals))}},
        "direction_scorer":    {"direction_correct":   {"mean": float(np.mean(dir_vals))}},
        "induced_fit_scorer":  {"induced_fit_correct": {"mean": float(np.mean(if_vals))}},
        "signed_error_scorer": {"signed_error_kcal":   {"mean": float(np.mean(bias_vals))}},
    }

    mae     = results["mae_scorer"]["mae_kcal_mol"]["mean"]
    dir_acc = results["direction_scorer"]["direction_correct"]["mean"]
    if_acc  = results["induced_fit_scorer"]["induced_fit_correct"]["mean"]
    bias    = results["signed_error_scorer"]["signed_error_kcal"]["mean"]

    print(f"\n  ── {label} Results ──────────────────────────────────")
    print(f"  MAE:                 {mae:.3f} kcal/mol")
    print(f"  Direction accuracy:  {100*dir_acc:.1f}%")
    print(f"  Induced-fit acc.:    {100*if_acc:.1f}%")
    print(f"  Mean signed error:   {bias:+.3f} kcal/mol")
    print()

    return results


def run_local_traces() -> None:
    """Deep-dive traces for five interesting poses (local mode)."""
    print("\n[Trace] Deep-dive traces for interesting poses:\n")
    interesting_poses = [
        (0.0,    0.0,  "far — no interaction"),
        (0.5,    0.0,  "mid-approach — partial contact"),
        (1.0,    0.0,  "fully docked — ideal orientation"),
        (1.0,   90.0,  "fully docked — 90° rotation"),
        (1.0,  180.0,  "fully docked — flipped 180°"),
    ]

    header = f"  {'Pose':<36} {'physics':>9}  {'nn':>9}  {'error':>7}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for approach_t, rotate_deg, desc in interesting_poses:
        drug = make_drug_atoms()
        place_drug(drug, approach_t, rotate_deg)

        ph_result = _physics.score(drug, _protein)
        ph_e      = float(np.clip(ph_result['total'], -50.0, 50.0))

        if _scorer is not None:
            nn_e  = float(_scorer.predict(drug, _protein))
            err   = nn_e - ph_e
            nn_s  = f"{nn_e:+7.2f}"
            err_s = f"{err:+.2f}"
        else:
            nn_s  = "    N/A"
            err_s = "   N/A"

        print(f"  {desc:<36} {ph_e:+9.2f}  {nn_s}  {err_s}")


# ═══════════════════════════════════════════════════════════════════════════
# §4  WEAVE MODELS  (only imported / used when Weave is active)
# ═══════════════════════════════════════════════════════════════════════════

def _build_weave_models_and_scorers():
    """
    Import weave and define models + scorers.
    Deferred so --local mode never touches weave at all.
    """
    import weave  # noqa: PLC0415

    class NNBindingModel(weave.Model):
        """
        Neural-network binding energy scorer.
        Predicts ΔG using the trained MLP (130-d pairwise distance features).
        """
        model_name: str = "MLP_130d_v1"

        @weave.op()
        def predict(self, approach_t: float, rotate_deg: float) -> dict:
            if _scorer is None:
                raise RuntimeError(
                    "No trained model found. Run python train.py first.")
            drug = make_drug_atoms()
            place_drug(drug, approach_t, rotate_deg)
            energy = _scorer.predict(drug, _protein)
            return {
                "energy":      round(float(energy), 3),
                "favorable":   bool(energy < 0.0),
                "induced_fit": bool(energy < INDUCED_FIT_THRESHOLD),
                "scorer":      "nn",
            }

    class PhysicsBaselineModel(weave.Model):
        """
        Classical physics-engine baseline (should score MAE ≈ 0).
        Treated as a model so Weave can compare it against the NN.
        """
        model_name: str = "PhysicsEngine_baseline"

        @weave.op()
        def predict(self, approach_t: float, rotate_deg: float) -> dict:
            drug   = make_drug_atoms()
            place_drug(drug, approach_t, rotate_deg)
            result = _physics.score(drug, _protein)
            energy = float(np.clip(result['total'], -50.0, 50.0))
            return {
                "energy":      round(energy, 3),
                "favorable":   bool(energy < 0.0),
                "induced_fit": bool(energy < INDUCED_FIT_THRESHOLD),
                "scorer":      "physics",
            }

    class RandomBaselineModel(weave.Model):
        """
        Random baseline — uniform energy ∈ [-20, 50] kcal/mol.
        Expected MAE ≈ 20–25. Acts as a sanity-check lower bound.
        """
        model_name: str = "Random_baseline"
        seed: int = 42

        @weave.op()
        def predict(self, approach_t: float, rotate_deg: float) -> dict:
            rng    = np.random.default_rng(
                self.seed + int(approach_t * 1000) + int(rotate_deg))
            energy = float(rng.uniform(-20.0, 50.0))
            return {
                "energy":      round(energy, 3),
                "favorable":   bool(energy < 0.0),
                "induced_fit": bool(energy < INDUCED_FIT_THRESHOLD),
                "scorer":      "random",
            }

    # ── Scorers ────────────────────────────────────────────────────────────

    @weave.op()
    def mae_scorer(physics_energy: float, output: dict) -> dict:
        """Mean Absolute Error vs physics ground truth (kcal/mol)."""
        return {"mae_kcal_mol": round(abs(output["energy"] - physics_energy), 3)}

    @weave.op()
    def direction_scorer(label_favorable: bool, output: dict) -> dict:
        """Binary: does the model correctly predict favorable vs. unfavorable?"""
        return {"direction_correct": int(output["favorable"] == label_favorable)}

    @weave.op()
    def induced_fit_scorer(label_induced_fit: bool, output: dict) -> dict:
        """Binary: does the model correctly predict induced-fit threshold crossing?"""
        return {"induced_fit_correct": int(output["induced_fit"] == label_induced_fit)}

    @weave.op()
    def signed_error_scorer(physics_energy: float, output: dict) -> dict:
        """Signed error (predicted − actual) — reveals systematic bias."""
        return {"signed_error_kcal": round(output["energy"] - physics_energy, 3)}

    @weave.op()
    def trace_single_pose(approach_t: float, rotate_deg: float) -> dict:
        """
        Detailed trace for one drug pose — logs physics breakdown alongside
        NN prediction.  Call this manually to inspect a specific configuration.

        Example:
            result = trace_single_pose(approach_t=1.0, rotate_deg=0)
        """
        drug = make_drug_atoms()
        place_drug(drug, approach_t, rotate_deg)
        ph    = _physics.score(drug, _protein)
        nn_e  = _scorer.predict(drug, _protein) if _scorer else None
        return {
            "approach_t":  approach_t,
            "rotate_deg":  rotate_deg,
            "physics": {
                "total":         round(ph["total"], 3),
                "vdw":           round(ph["vdw"], 3),
                "electrostatic": round(ph["electrostatic"], 3),
                "hbonds":        round(ph["hbonds"], 3),
                "hydrophobic":   round(ph["hydrophobic"], 3),
                "n_hbonds":      ph["n_hbonds"],
            },
            "nn_predicted":          round(nn_e, 3) if nn_e is not None else None,
            "nn_error":              round(nn_e - ph["total"], 3) if nn_e is not None else None,
            "induced_fit_triggered": ph["total"] < INDUCED_FIT_THRESHOLD,
        }

    return (
        NNBindingModel, PhysicsBaselineModel, RandomBaselineModel,
        mae_scorer, direction_scorer, induced_fit_scorer,
        signed_error_scorer, trace_single_pose,
    )


# ═══════════════════════════════════════════════════════════════════════════
# §5  WEAVE EVALUATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════

async def run_weave_evaluation(model, dataset: list[dict], label: str,
                               scorers: list) -> dict:
    """Runs weave.Evaluation for one model and prints a summary."""
    import weave  # noqa: PLC0415

    print(f"\n[Eval] Running Weave evaluation for: {label}")

    evaluation = weave.Evaluation(
        name        = f"binding_energy_eval_{label}",
        description = (
            f"Evaluates {label} on 72 diverse drug poses. "
            f"Scores: MAE, direction accuracy, induced-fit accuracy, signed error."
        ),
        dataset     = dataset,
        scorers     = scorers,
    )

    results = await evaluation.evaluate(model)

    # ── Print summary ─────────────────────────────────────────────────────
    def _mean(key, subkey):
        try:
            return results[key][subkey]["mean"]
        except (KeyError, TypeError):
            return None

    mae     = _mean("mae_scorer",          "mae_kcal_mol")
    dir_acc = _mean("direction_scorer",    "direction_correct")
    if_acc  = _mean("induced_fit_scorer",  "induced_fit_correct")
    bias    = _mean("signed_error_scorer", "signed_error_kcal")

    print(f"\n  ── {label} Results ──────────────────────────────────")
    if mae     is not None: print(f"  MAE:                 {mae:.3f} kcal/mol")
    if dir_acc is not None: print(f"  Direction accuracy:  {100*dir_acc:.1f}%")
    if if_acc  is not None: print(f"  Induced-fit acc.:    {100*if_acc:.1f}%")
    if bias    is not None: print(f"  Mean signed error:   {bias:+.3f} kcal/mol")
    print()

    return results


# ═══════════════════════════════════════════════════════════════════════════
# §6  ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════

def local_main(dataset: list[dict]) -> None:
    """Full evaluation without any Weave / W&B dependency."""
    models = [
        ("nn",      "nn_mlp"),
        ("physics", "physics_baseline"),
        ("random",  "random_baseline"),
    ]

    all_results = {}
    for mode, label in models:
        all_results[label] = run_local_evaluation(dataset, mode, label)

    run_local_traces()

    print("\n" + "═" * 60)
    print("  LOCAL EVALUATION COMPLETE")
    print("  All metrics computed — no W&B login required.")
    print()
    print("  To enable full Weave tracing (per-example traces,")
    print("  side-by-side comparisons, interactive dashboard):")
    print("    1. wandb login        # paste key from wandb.ai/authorize")
    print("    2. python weave_eval.py")
    print("═" * 60 + "\n")


async def weave_main(dataset: list[dict]) -> None:
    """Full evaluation with Weave tracing (requires W&B login)."""
    import weave  # noqa: PLC0415

    weave.init("protein-drug-binding")

    (
        NNBindingModel, PhysicsBaselineModel, RandomBaselineModel,
        mae_scorer, direction_scorer, induced_fit_scorer,
        signed_error_scorer, trace_single_pose,
    ) = _build_weave_models_and_scorers()

    # ── Publish dataset as a Weave artifact ─────────────────────────────
    published = weave.publish(
        weave.Dataset(name="docking_poses_72", rows=dataset),
    )
    print(f"[Weave] Dataset published: {published.uri()}")

    scorers = [mae_scorer, direction_scorer, induced_fit_scorer, signed_error_scorer]

    models = [
        (NNBindingModel(),       "nn_mlp"),
        (PhysicsBaselineModel(), "physics_baseline"),
        (RandomBaselineModel(),  "random_baseline"),
    ]

    all_results = {}
    for model, label in models:
        all_results[label] = await run_weave_evaluation(model, dataset, label, scorers)

    # ── Single-pose deep dives ───────────────────────────────────────────
    print("\n[Trace] Deep-dive traces for interesting poses:\n")
    interesting_poses = [
        (0.0,    0.0,  "far — no interaction"),
        (0.5,    0.0,  "mid-approach — partial contact"),
        (1.0,    0.0,  "fully docked — ideal orientation"),
        (1.0,   90.0,  "fully docked — 90° rotation"),
        (1.0,  180.0,  "fully docked — flipped 180°"),
    ]
    for approach_t, rotate_deg, desc in interesting_poses:
        result = trace_single_pose(approach_t, rotate_deg)
        nn_e   = result["nn_predicted"]
        ph_e   = result["physics"]["total"]
        err    = result["nn_error"]
        print(f"  {desc:<35}  physics={ph_e:+7.2f}  "
              f"nn={nn_e:+7.2f}  err={err:+.2f}")

    # ── Final summary ────────────────────────────────────────────────────
    entity = os.environ.get("WANDB_ENTITY", "<your-username>")
    print("\n" + "═" * 60)
    print("  WEAVE EVALUATION COMPLETE")
    print("  View full traces and metrics at:")
    print(f"  https://wandb.ai/{entity}/protein-drug-binding")
    print("  (Click the 'Weave' tab in the W&B project)")
    print("═" * 60 + "\n")


def main() -> None:
    args = _parse_args()

    # ── Build dataset (shared for both modes) ────────────────────────────
    dataset = build_test_dataset()

    if args.local:
        local_main(dataset)
    else:
        asyncio.run(weave_main(dataset))


if __name__ == "__main__":
    main()
