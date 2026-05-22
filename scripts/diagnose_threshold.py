#!/usr/bin/env python3
"""
Diagnose how anomaly_ratio affects threshold and predictions.

Usage:
    python scripts/diagnose_threshold.py \
        --dataset CUSTOM \
        --data_path ./data/ \
        --win_size 100 \
        --input_c 33 \
        --batch_size 8 \
        --n_memory 128 \
        --d_model 512 \
        --device cuda:0 \
        --model_save_path checkpoints
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from solver import Solver
from model.loss_functions import GatheringLoss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CUSTOM")
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--input_c", type=int, default=33)
    parser.add_argument("--output_c", type=int, default=33)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--win_size", type=int, default=100)
    parser.add_argument("--n_memory", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_save_path", type=str, default="checkpoints")
    parser.add_argument("--anomaly_ratios", type=float, nargs='+', default=[0.5, 1.0, 2.0, 5.0, 10.0])
    args = parser.parse_args()

    config = vars(args)
    config.update({
        "lr": 0.0001, "temperature": 0.1, "lambd": 0.01,
        "temp_param": 0.05, "num_epochs": 100, "k": 5,
        "pretrained_model": None, "n_head": 8,
        "n_enc_layers": 1, "n_dec_layers": 1,
        "dropout": 0.1, "entropy_param": 0.05, "gamma": 0.01,
        "mode": "test", "memory_initial": "False", "phase_type": "test",
        "data_path": args.data_path, "anomaly_ratio": 1.0,  # dummy, will be overwritten
    })

    solver = Solver(config)

    # Load checkpoint — saved with DataParallel (module. prefix), so strip it
    ckpt_path = os.path.join(args.model_save_path, f"{args.dataset}_checkpoint_second_train.pth")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt_stripped = {k.replace('module.', ''): v for k, v in ckpt.items()}
    solver.model.load_state_dict(ckpt_stripped, strict=False)
    solver.model.eval()

    criterion = torch.nn.MSELoss(reduce=False)
    gathering_loss = GatheringLoss(reduce=False)
    temperature = solver.temperature

    # Compute test energy for ALL test windows
    all_energy = []
    all_labels = []
    with torch.no_grad():
        for i, (input_data, labels) in enumerate(solver.test_loader):
            input = input_data.float().to(solver.device)
            output_dict = solver.model(input)
            output, queries, mem_items = output_dict['out'], output_dict['queries'], output_dict['mem']

            rec_loss = torch.mean(criterion(input, output), dim=-1)
            latent_score = torch.softmax(gathering_loss(queries, mem_items) / temperature, dim=-1)
            loss = latent_score * rec_loss

            all_energy.append(loss.detach().cpu().numpy().flatten())
            all_labels.append(labels.flatten())

    test_energy = np.concatenate(all_energy, axis=0)
    test_labels = np.concatenate(all_labels, axis=0).astype(int)

    n_anomaly = test_labels.sum()
    n_total = len(test_labels)
    print(f"Test samples: {n_total}, Anomalies: {n_anomaly} ({100*n_anomaly/n_total:.1f}%)")
    print(f"Energy stats: min={test_energy.min():.6f} max={test_energy.max():.6f} "
          f"mean={test_energy.mean():.6f} median={np.median(test_energy):.6f}")
    print()

    # Evaluate at different anomaly_ratio values
    print(f"{'Ratio':>6} | {'Threshold':>10} | {'Pred+':>6} | {'TP':>4} | {'FP':>4} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
    print("-" * 80)

    results = []
    for ratio in sorted(args.anomaly_ratios):
        thresh = np.percentile(test_energy, 100 - ratio)
        pred = (test_energy > thresh).astype(int)

        tp = int(((pred == 1) & (test_labels == 1)).sum())
        fp = int(((pred == 1) & (test_labels == 0)).sum())
        fn = int(((pred == 0) & (test_labels == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            'ratio': ratio, 'threshold': thresh, 'pred_pos': pred.sum(),
            'tp': tp, 'fp': fp, 'precision': precision, 'recall': recall, 'f1': f1
        })
        print(f"{ratio:>6.1f} | {thresh:>10.6f} | {pred.sum():>6} | {tp:>4} | {fp:>4} | {precision:>9.4f} | {recall:>6.4f} | {f1:>6.4f}")

    # Plot energy distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of energy (log scale)
    ax = axes[0]
    ax.hist(test_energy, bins=100, color='steelblue', alpha=0.7, edgecolor='white')
    for r in results:
        ax.axvline(r['threshold'], color='red', linestyle='--', alpha=0.7,
                   label=f"ratio={r['ratio']:.1f}  thr={r['threshold']:.4f}")
    ax.set_xlabel('Energy')
    ax.set_ylabel('Count')
    ax.set_title('Energy Score Distribution (test set)')
    ax.legend(fontsize=7)
    ax.set_xscale('log')

    # Right: Precision, Recall, F1 vs anomaly_ratio
    ax = axes[1]
    ratios = [r['ratio'] for r in results]
    ax.plot(ratios, [r['precision'] for r in results], 'o-', label='Precision', color='blue')
    ax.plot(ratios, [r['recall'] for r in results], 's-', label='Recall', color='green')
    ax.plot(ratios, [r['f1'] for r in results], '^-', label='F1', color='red')
    ax.set_xlabel('anomaly_ratio')
    ax.set_ylabel('Score')
    ax.set_title('Metrics vs anomaly_ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PROJECT_ROOT, "png", "threshold_diagnosis.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")


if __name__ == "__main__":
    main()
