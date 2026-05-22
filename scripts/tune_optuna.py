#!/usr/bin/env python3
"""
MEMTO Hyperparameter Tuning with Optuna.

Usage:
    pip install optuna
    python scripts/tune_optuna.py

Search space & number of trials can be adjusted below.
"""

import os
import sys
import argparse
import torch
import numpy as np
import optuna
import csv
import datetime

# Add project root to path so we can import solver & utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from solver import Solver
from torch.backends import cudnn


# ──────────────────────────────────────────────
# Search space definition
# ──────────────────────────────────────────────
def define_search_space(trial):
    """Return a dict of hyperparameters sampled from the search space."""
    return {
        # Core hyperparams (most impactful)
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "d_model": trial.suggest_categorical("d_model", [128, 256, 512]),
        "n_memory": trial.suggest_categorical("n_memory", [32, 64, 128, 256]),
        "temperature": trial.suggest_float("temperature", 0.01, 0.5, log=True),
        "lambd": trial.suggest_float("lambd", 1e-3, 0.1, log=True),
        "temp_param": trial.suggest_float("temp_param", 0.01, 0.2, log=True),
        # Secondary hyperparams
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
        "num_epochs": trial.suggest_int("num_epochs", 5, 20),
        "win_size": trial.suggest_categorical("win_size", [50, 100, 200]),
        "anomaly_ratio": trial.suggest_float("anomaly_ratio", 0.5, 2.0),
    }


# ──────────────────────────────────────────────
# Objective function
# ──────────────────────────────────────────────
def run_trial(trial, config_base):
    """
    Run the full MEMTO pipeline (Phase 1 → K-means → Phase 2 → Test)
    and return the F1 score to maximize.
    """
    hparams = define_search_space(trial)

    # Build trial-specific config
    config = argparse.Namespace(**config_base)
    for k, v in hparams.items():
        setattr(config, k, v)

    # Derive output_c from input_c
    config.output_c = config.input_c

    # Isolated checkpoint & memory_item paths per trial to avoid overwriting
    trial_id = f"trial_{trial.number}"
    config.model_save_path = os.path.join(PROJECT_ROOT, "checkpoints", f"optuna_{trial_id}")
    os.makedirs(config.model_save_path, exist_ok=True)

    cudnn.benchmark = True

    # ── Phase 1: First training (random memory init) ──
    print(f"\n{'='*60}")
    print(f"  TRIAL {trial.number} — Phase 1 (random memory)")
    print(f"  LR={hparams['lr']:.6f}  D_MODEL={hparams['d_model']}  N_MEM={hparams['n_memory']}")
    print(f"  TEMP={hparams['temperature']}  LAMBDA={hparams['lambd']}")
    print(f"{'='*60}\n")

    solver = Solver(vars(config))
    solver.train(training_type='first_train')

    # ── Phase 2: Memory initialization (K-means) ──
    print(f"\n  TRIAL {trial.number} — Phase 2 (K-means)")
    solver.get_memory_initial_embedding(training_type='second_train')

    # ── Phase 3: Second training (K-means initialized memory) ──
    print(f"\n  TRIAL {trial.number} — Phase 3 (K-means memory training)")
    # Create a fresh solver so model gets reloaded properly
    solver2 = Solver(vars(config))
    solver2.train(training_type='second_train')

    # ── Phase 4: Test ──
    print(f"\n  TRIAL {trial.number} — Phase 4 (Test)")
    accuracy, precision, recall, f_score, auc_pr, rp, rr, rf = solver2.test()

    return f_score


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MEMTO Optuna hyperparameter tuning")

    # Dataset args
    parser.add_argument("--dataset", type=str, default="CUSTOM")
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--input_c", type=int, default=33)
    parser.add_argument("--anomaly_ratio", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)

    # Optuna args
    parser.add_argument("--n_trials", type=int, default=30, help="number of trials to run")
    parser.add_argument("--direction", type=str, default="maximize", choices=["maximize", "minimize"])
    parser.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random", "cmaes"],
                        help="Optuna sampler")
    parser.add_argument("--pruner", type=str, default="none", choices=["median", "hyperband", "none"],
                        help="Optuna pruner (use 'none' since we report final F1 only)")
    parser.add_argument("--study_name", type=str, default="memto_tuning")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g. sqlite:///optuna.db)")
    parser.add_argument("--load_if_exists", action="store_true",
                        help="resume an existing study instead of creating a new one")
    parser.add_argument("--results_csv", type=str, default="scripts/tuning_results.csv",
                        help="path to save per-trial results CSV")

    args = parser.parse_args()

    # Build base config (fixed params that won't be tuned)
    config_base = {
        "dataset": args.dataset,
        "data_path": args.data_path,
        "input_c": args.input_c,
        "output_c": args.input_c,
        "anomaly_ratio": args.anomaly_ratio,
        "device": args.device,
        "num_workers": args.num_workers,
        # Fixed defaults (overridden by search space during trials)
        "lr": 0.0001,
        "d_model": 512,
        "n_memory": 128,
        "temperature": 0.1,
        "lambd": 0.01,
        "temp_param": 0.05,
        "batch_size": 8,
        "num_epochs": 10,
        "win_size": 100,
        "k": 5,
        "pretrained_model": None,
        "model_save_path": "checkpoints",
        "n_head": 8,
        "n_enc_layers": 1,
        "n_dec_layers": 1,
        "dropout": 0.1,
        "entropy_param": 0.05,
        "gamma": 0.01,
        "mode": "train",
        "memory_initial": "False",   # MUST be string — solver compares with == "False"
        "phase_type": None,
    }

    # Set up sampler and pruner
    sampler_map = {
        "tpe": optuna.samplers.TPESampler(seed=42),
        "random": optuna.samplers.RandomSampler(seed=42),
        "cmaes": optuna.samplers.CmaEsSampler(seed=42),
    }
    sampler = sampler_map[args.sampler]

    pruner_map = {
        "median": optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        "hyperband": optuna.pruners.HyperbandPruner(min_resource=3, max_resource=20, reduction_factor=3),
        "none": optuna.pruners.NopPruner(),
    }
    pruner = pruner_map[args.pruner]

    # Create or resume study
    study = optuna.create_study(
        study_name=args.study_name,
        direction=args.direction,
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=args.load_if_exists,
    )

    print(f"Starting Optuna study: {args.study_name}")
    print(f"  Sampler: {args.sampler}, Pruner: {args.pruner}")
    print(f"  Trials: {args.n_trials}, Direction: {args.direction}")
    print(f"  Storage: {args.storage or 'in-memory'}")

    # Callback to save results per trial
    os.makedirs(os.path.dirname(os.path.abspath(args.results_csv)), exist_ok=True)
    csv_header = None

    def callback(study, trial):
        nonlocal csv_header
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        values = trial.params.copy()
        values["trial_number"] = trial.number
        values["f1_score"] = trial.value
        values["best_f1_so_far"] = study.best_trial.value if study.best_trial else trial.value
        if csv_header is None:
            csv_header = list(values.keys())
            with open(args.results_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_header)
                writer.writeheader()
                writer.writerow(values)
        else:
            with open(args.results_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_header)
                writer.writerow(values)

    study.optimize(
        lambda trial: run_trial(trial, config_base),
        n_trials=args.n_trials,
        callbacks=[callback],
    )

    print(f"\n{'='*60}")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best F1:    {study.best_trial.value:.4f}")
    print(f"  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.results_csv}")


if __name__ == "__main__":
    main()
