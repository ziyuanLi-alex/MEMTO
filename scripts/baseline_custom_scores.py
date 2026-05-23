#!/usr/bin/env python3
"""
Data-level baseline diagnostics for CUSTOM dataset.

No model training required — reads data/train.csv directly, applies the same
80/20 split as CustomCSVSegLoader, fits a scaler on normal-only training data,
and computes anomaly scores using five feature-level methods:

  1. feature z-score:  max|z| per window  /  L2 norm of z
  2. first-diff z-score:  max|z_diff| per window
  3. rolling std / low-variance score
  4. PCA reconstruction error  (fit on normal train, evaluate on test)
  5. IsolationForest  (fit on normal train, predict on test)

Also produces time-series plots: label, feature_norm, diff_norm, rolling_std.

If these baselines achieve AUC-PR near the MEMTO result (~0.028), the problem
is in the data/labels/split/features.  If they do better (e.g. > 0.1), MEMTO
is simply not adapting to a signal that already exists.
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ──────────────────────────────────────────────────────────────
#  Data loading — mirrors CustomCSVSegLoader logic
# ──────────────────────────────────────────────────────────────

def load_and_split(data_path, win_size=100, step=100, val_ratio=0.2, use_iqr=True):
    """
    Returns (train_data, train_labels, test_data, test_labels, scaler)
    where train_data/test_data are already scaled (shape: (N_timesteps, n_features)).
    Labels are per-timestep binary arrays.
    """
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    feature_cols = [c for c in train_df.columns if c != "y"]
    raw_features = train_df[feature_cols].values.astype(np.float64)
    all_labels = train_df["y"].values.astype(np.int64)

    split_idx = int(len(raw_features) * (1 - val_ratio))

    raw_train = raw_features[:split_idx]
    raw_test = raw_features[split_idx:]
    train_labels = all_labels[:split_idx]
    test_labels = all_labels[split_idx:]

    # IQR clipping on raw train (before scaler fit)
    lower_bound, upper_bound = None, None
    if use_iqr:
        q1 = np.percentile(raw_train, 25, axis=0)
        q3 = np.percentile(raw_train, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        raw_train = np.clip(raw_train, lower_bound, upper_bound)
        if lower_bound is not None:
            raw_test = np.clip(raw_test, lower_bound, upper_bound)

    scaler = StandardScaler()
    normal_mask = train_labels == 0
    scaler.fit(raw_train[normal_mask])

    # Guard against zero-variance features (std=0 → NaN after transform)
    train_scaled = np.nan_to_num(scaler.transform(raw_train), nan=0.0)
    test_scaled = np.nan_to_num(scaler.transform(raw_test), nan=0.0)

    return train_scaled, train_labels, test_scaled, test_labels


# ──────────────────────────────────────────────────────────────
#  Sliding-window helpers
# ──────────────────────────────────────────────────────────────

def sliding_window_starts(n, win_size, step, labels=None, normal_only=False):
    starts = []
    for s in range(0, n - win_size + 1, step):
        if normal_only and labels is not None and labels[s:s + win_size].max() > 0:
            continue
        starts.append(s)
    return np.array(starts, dtype=np.int64)


def window_score(per_timestep, starts, win_size, agg="max"):
    """Reduce per-timestep scores to per-window scores."""
    out = np.empty(len(starts), dtype=np.float64)
    for i, s in enumerate(starts):
        window = per_timestep[s:s + win_size]
        if agg == "max":
            out[i] = window.max()
        elif agg == "mean":
            out[i] = window.mean()
        elif agg == "sum":
            out[i] = window.sum()
        else:
            raise ValueError(agg)
    return out


# ──────────────────────────────────────────────────────────────
#  Baseline methods
# ──────────────────────────────────────────────────────────────

def baseline_zscore_max(train_data, test_data, starts_test):
    """Per-window max |z-score| (higher = more anomalous)."""
    test_z = np.abs(test_data)
    return window_score(test_z, starts_test, win_size=100, agg="max")


def baseline_zscore_l2(train_data, test_data, starts_test):
    """Per-window L2 norm of z-score vector."""
    out = np.empty(len(starts_test), dtype=np.float64)
    for i, s in enumerate(starts_test):
        window = test_data[s:s + 100]
        out[i] = np.linalg.norm(window, axis=1).mean()
    return out


def baseline_diff_zscore(train_data, test_data, starts_test):
    """Per-window max |first-diff| in z-space."""
    test_diff = np.diff(test_data, axis=0)
    test_diff = np.abs(test_diff)
    # Diff has one fewer row; adjust starts by 1
    starts_adj = np.clip(starts_test, 1, len(test_data) - 2)
    return window_score(test_diff, starts_adj, win_size=99, agg="max")


def baseline_rolling_std(test_data, starts_test, roll_window=20):
    """
    Per-window mean rolling-std (low-variance anomaly):
    compute rolling std with a short window, then take mean per long window.
    Anomalies can be EITHER too smooth (low rolling std) or too volatile.
    We return 1 / (rolling_std + eps) so that smooth = high score.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    rolled = sliding_window_view(test_data, (roll_window, test_data.shape[1]))
    # rolled shape: (N-roll+1, 1, n_feat) — wait, let's do it per-feature
    # Actually simpler: compute std over axis=1 with a manual loop
    n = len(test_data)
    n_feat = test_data.shape[1]
    std_arr = np.full(n, np.nan, dtype=np.float64)
    for i in range(n - roll_window + 1):
        std_arr[i + roll_window // 2] = np.std(test_data[i:i + roll_window], axis=0).mean()

    # Fill edges
    valid = std_arr[~np.isnan(std_arr)]
    fill_val = valid.mean() if len(valid) > 0 else 0.0
    std_arr = np.nan_to_num(std_arr, nan=fill_val)

    # Low-variance score: 1 / (std + eps)
    low_var_score = 1.0 / (std_arr + 1e-8)

    return window_score(low_var_score, starts_test, win_size=100, agg="mean")


def baseline_pca_reconstruction(train_data, test_data, starts_test, n_components=10):
    """
    Fit PCA on normal train data, reconstruct test windows, return MSE per window.
    """
    pca = PCA(n_components=n_components)
    pca.fit(train_data)
    reconstructed = pca.inverse_transform(pca.transform(test_data))
    per_timestep_mse = np.mean((test_data - reconstructed) ** 2, axis=1)
    return window_score(per_timestep_mse, starts_test, win_size=100, agg="mean")


def baseline_isolation_forest(train_data, test_data, starts_test, contamination=0.01):
    """
    Fit IF on normal train data, get anomaly scores on test.
    sklearn IF: negative score = normal, positive = anomalous.
    We negate so that higher = more anomalous.
    """
    rng = np.random.RandomState(42)
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=rng,
        max_samples=min(256, len(train_data)),
    )
    clf.fit(train_data)
    per_timestep_score = -clf.score_samples(test_data)  # negate: higher = anomalous
    return window_score(per_timestep_score, starts_test, win_size=100, agg="mean")


# ──────────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate(scores, window_labels, name):
    """Print AUC-PR and AUC-ROC for one method."""
    auc_pr = average_precision_score(window_labels, scores)
    auc_roc = roc_auc_score(window_labels, scores)
    print(f"  {name:30s}  AUC-PR={auc_pr:.4f}  AUC-ROC={auc_roc:.4f}")
    return auc_pr, auc_roc


def make_window_labels(test_labels, starts_test, win_size=100):
    """A window is anomalous if any timestep in it is anomalous."""
    labels = np.zeros(len(starts_test), dtype=np.int64)
    for i, s in enumerate(starts_test):
        labels[i] = int(test_labels[s:s + win_size].max() > 0)
    return labels


# ──────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────

def plot_diagnostics(test_data, test_labels, starts_test,
                     scores_dict, out_path):
    """
    Time-series plots over the test portion:
    - labels (top)
    - feature_norm  (L2 norm of z-score per timestep)
    - diff_norm     (L2 norm of first-diff)
    - rolling_std   (mean rolling std)
    """
    n = len(test_data)
    feature_norm = np.linalg.norm(test_data, axis=1)
    diff_norm = np.linalg.norm(np.diff(test_data, axis=0), axis=1)
    diff_norm = np.concatenate([[diff_norm[0]], diff_norm])  # pad to same length

    roll_window = 20
    from numpy.lib.stride_tricks import sliding_window_view
    rolled_std = np.full(n, np.nan, dtype=np.float64)
    for i in range(n - roll_window + 1):
        rolled_std[i + roll_window // 2] = np.std(test_data[i:i + roll_window], axis=0).mean()
    rolled_std = np.nan_to_num(rolled_std, nan=np.nanmean(rolled_std))

    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    x = np.arange(n)

    # 1. Labels
    axes[0].fill_between(x, 0, 1, where=test_labels > 0, alpha=0.6, color='red', step='mid')
    axes[0].set_ylabel("Label (y)")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_title("Test segment — labels")
    axes[0].axhline(0.5, color='gray', lw=0.5)

    # 2. Feature norm
    axes[1].plot(x, feature_norm, lw=0.5, color='blue', alpha=0.8)
    axes[1].set_ylabel("Feature L2 norm")
    axes[1].set_title("Scaled feature L2 norm per timestep")
    axes[1].axhline(feature_norm[test_labels == 0].mean(), color='green', ls='--',
                     label='Normal mean', lw=1)
    axes[1].axhline(feature_norm[test_labels == 1].mean(), color='red', ls='--',
                     label='Anomaly mean', lw=1)
    axes[1].legend(loc='upper right')

    # 3. Diff norm
    axes[2].plot(x, diff_norm, lw=0.5, color='orange', alpha=0.8)
    axes[2].set_ylabel("Diff L2 norm")
    axes[2].set_title("First-diff L2 norm per timestep")

    # 4. Rolling std
    axes[3].plot(x, rolled_std, lw=0.5, color='purple', alpha=0.8)
    axes[3].set_ylabel("Rolling std")
    axes[3].set_title(f"Rolling std (window={roll_window})")

    # 5. Baseline scores overlay
    ax5 = axes[4]
    colors = plt.cm.tab10(np.linspace(0, 1, len(scores_dict)))
    for j, (name, scores) in enumerate(scores_dict.items()):
        # Upsample window scores to per-timestep for plotting
        per_ts = np.zeros(n, dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)
        for i, s in enumerate(starts_test):
            per_ts[s:s + 100] += scores[i]
            counts[s:s + 100] += 1
        per_ts = np.where(counts > 0, per_ts / counts, 0)
        # Normalize to [0,1]
        s_min, s_max = per_ts.min(), per_ts.max()
        if s_max > s_min:
            per_ts = (per_ts - s_min) / (s_max - s_min)
        ax5.plot(x, per_ts, lw=0.7, alpha=0.7, color=colors[j], label=name)
    ax5.set_ylabel("Baseline score (normed)")
    ax5.set_title("Baseline anomaly scores (normalized, overlaid)")
    ax5.legend(loc='upper right', fontsize=7)

    for ax in axes:
        ax.grid(True, alpha=0.2)

    plt.tight_layout(h_pad=0.5)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Baseline anomaly scores for CUSTOM dataset")
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--win_size", type=int, default=100)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--use_iqr", type=int, default=1)
    parser.add_argument("--n_pca_components", type=int, default=10)
    parser.add_argument("--if_contamination", type=float, default=0.01)
    args = parser.parse_args()

    print("=" * 60)
    print("  BASELINE CUSTOM SCORES — data-level anomaly detection")
    print("=" * 60)

    # ── Load ──
    train_data, train_labels, test_data, test_labels = load_and_split(
        args.data_path,
        win_size=args.win_size,
        step=args.step,
        val_ratio=args.val_ratio,
        use_iqr=args.use_iqr,
    )
    n_train, n_test = len(train_data), len(test_data)
    n_feat = train_data.shape[1]
    print(f"\nTrain: {n_train} timesteps,  Test: {n_test} timesteps,  Features: {n_feat}")
    print(f"  Train anomaly ratio: {100 * train_labels.mean():.1f}%")
    print(f"  Test  anomaly ratio: {100 * test_labels.mean():.1f}%")

    # ── Window starts (test, no normal_only) ──
    starts_test = sliding_window_starts(n_test, args.win_size, args.step)
    window_labels = make_window_labels(test_labels, starts_test)
    print(f"  Test windows: {len(starts_test)}, anomaly windows: {window_labels.sum()}")
    print(f"  Window anomaly ratio: {100 * window_labels.mean():.1f}%")

    # ── Compute baselines ──
    print("\nComputing baseline scores...")

    scores_z_max = baseline_zscore_max(train_data, test_data, starts_test)
    scores_z_l2 = baseline_zscore_l2(train_data, test_data, starts_test)
    scores_diff = baseline_diff_zscore(train_data, test_data, starts_test)
    scores_roll = baseline_rolling_std(test_data, starts_test)
    scores_pca = baseline_pca_reconstruction(
        train_data, test_data, starts_test, n_components=args.n_pca_components
    )
    scores_if = baseline_isolation_forest(
        train_data, test_data, starts_test, contamination=args.if_contamination
    )

    baseline_dict = {
        "z-score max": scores_z_max,
        "z-score L2": scores_z_l2,
        "diff z-score max": scores_diff,
        "rolling std (low-var)": scores_roll,
        "PCA recon error": scores_pca,
        "IsolationForest": scores_if,
    }

    # ── Evaluate ──
    print("\nResults (test windows):")
    print("-" * 60)
    results = {}
    for name, scores in baseline_dict.items():
        auc_pr, auc_roc = evaluate(scores, window_labels, name)
        results[name] = (auc_pr, auc_roc)

    best_pr_name = max(results, key=lambda k: results[k][0])
    best_pr = results[best_pr_name][0]
    print(f"\nBest baseline: {best_pr_name} (AUC-PR={best_pr:.4f})")

    # Interpretation
    print(f"\n{'=' * 60}")
    print("  INTERPRETATION:")
    if best_pr < 0.05:
        print("  All baselines are near random. The CUSTOM labels may not")
        print("  correspond to detectable numerical anomalies in the features.")
        print("  The anomaly might be semantic (rPPG pattern) rather than")
        print("  statistical (magnitude/variance change).")
    elif best_pr < 0.1:
        print("  Weak signal in baselines. MEMTO should do better than this")
        print("  if it's learning useful representations.  If MEMTO is at")
        print("  ~0.028 while baselines are ~0.05-0.08, the model is")
        print("  actively degrading the signal.")
    else:
        print(f"  Baselines show real signal (AUC-PR={best_pr:.4f}).")
        print("  If MEMTO is far below this, the model architecture or")
        print("  training is the bottleneck, not the data.")
    print(f"{'=' * 60}")

    # ── Plots ──
    plot_path = os.path.join(PROJECT_ROOT, "png", "baseline_diagnostics.png")
    plot_diagnostics(test_data, test_labels, starts_test, baseline_dict, plot_path)

    # ── Also plot label distribution over time ──
    fig, ax = plt.subplots(1, 1, figsize=(16, 3))
    x = np.arange(n_test)
    ax.fill_between(x, 0, 1, where=test_labels > 0, alpha=0.5, color='red', step='mid')
    ax.set_ylabel("Label")
    ax.set_title(f"Test labels over time (n={n_test}, anomaly={test_labels.sum()})")
    ax.set_xlabel("Timestep index (relative to test split)")
    ax.grid(True, alpha=0.2)
    label_path = os.path.join(PROJECT_ROOT, "png", "test_labels_timeline.png")
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    plt.savefig(label_path, dpi=150)
    print(f"Label timeline saved to: {label_path}")
    plt.close()


if __name__ == "__main__":
    main()
