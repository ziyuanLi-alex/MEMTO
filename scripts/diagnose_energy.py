#!/usr/bin/env python3
"""
Diagnostic: Decompose energy into rec_loss, latent_score, and their product.
Helps identify which component fails to distinguish anomalies.

Usage:
    # Default checkpoint
    python scripts/diagnose_energy.py

    # no-IQR checkpoint
    python scripts/diagnose_energy.py \
        --model_save_path checkpoints/exp_noiqr_lambd0 \
        --use_iqr 0
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.Transformer import TransformerVar
from model.loss_functions import GatheringLoss

from data_factory.data_loader import get_loader_segment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", type=str, default="checkpoints")
    parser.add_argument("--use_iqr", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="CUSTOM")
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--input_c", type=int, default=33)
    parser.add_argument("--output_c", type=int, default=33)
    parser.add_argument("--win_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_memory", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    DEVICE = args.device
    print(f"Model path: {args.model_save_path}")
    print(f"Use IQR:   {args.use_iqr}")
    print(f"Device:    {DEVICE}")
    print(f"n_memory:  {args.n_memory}, d_model: {args.d_model}")

    # ── Load data ──
    test_loader, _ = get_loader_segment(
        args.data_path, batch_size=args.batch_size,
        win_size=args.win_size, step=args.win_size,
        mode='test', dataset=args.dataset,
        use_iqr=args.use_iqr
    )

    # Also load train+val for IQR-based threshold (if use_iqr=1)
    if args.use_iqr:
        train_loader, _, _ = get_loader_segment(
            args.data_path, batch_size=args.batch_size,
            win_size=args.win_size, step=args.win_size,
            mode='train', dataset=args.dataset,
            use_iqr=args.use_iqr
        )

    # ── Load model ──
    ckpt_path = os.path.join(args.model_save_path, f"{args.dataset}_checkpoint_second_train.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_stripped = {k.replace('module.', ''): v for k, v in ckpt.items()}

    model = TransformerVar(
        win_size=args.win_size, enc_in=args.input_c, c_out=args.output_c,
        e_layers=3, d_model=args.d_model, n_memory=args.n_memory,
        device=DEVICE, memory_initial=False, memory_init_embedding=None,
        phase_type='test', dataset_name=args.dataset
    )
    model.load_state_dict(ckpt_stripped, strict=False)
    model.eval()
    model.to(DEVICE)

    criterion = torch.nn.MSELoss(reduction='none')
    gathering_loss = GatheringLoss(reduce=False)
    temperature = args.temperature

    # ── Collect scores ──
    all_rec_loss, all_latent_score, all_loss, all_labels = [], [], [], []

    with torch.no_grad():
        for input_data, labels in test_loader:
            input = input_data.float().to(DEVICE)
            output_dict = model(input)
            output = output_dict['out']
            queries = output_dict['queries']
            mem_items = output_dict['mem']

            rec_loss = torch.mean(criterion(input, output), dim=-1)  # (N, L)
            latent_score = torch.softmax(gathering_loss(queries, mem_items) / temperature, dim=-1)
            loss = latent_score * rec_loss

            all_rec_loss.append(rec_loss.cpu().numpy())
            all_latent_score.append(latent_score.cpu().numpy())
            all_loss.append(loss.cpu().numpy())
            all_labels.append(labels.reshape(-1, rec_loss.shape[-1]).numpy())

    rec_loss = np.concatenate(all_rec_loss, axis=0).flatten()
    latent_score = np.concatenate(all_latent_score, axis=0).flatten()
    energy = np.concatenate(all_loss, axis=0).flatten()
    labels = np.concatenate(all_labels, axis=0).flatten().astype(int)

    n_anomaly = labels.sum()
    n_normal = len(labels) - n_anomaly
    print(f"\nTest samples: {len(labels)}, Normal: {n_normal}, Anomaly: {n_anomaly}")
    print(f"  Anomaly ratio: {100*n_anomaly/len(labels):.1f}%")

    # ── AUC-PR ──
    from sklearn.metrics import average_precision_score
    auc_rec = average_precision_score(labels, rec_loss)
    auc_score = average_precision_score(labels, latent_score)
    auc_energy = average_precision_score(labels, energy)

    print(f"\nAUC-PR (higher = better separation):")
    print(f"  rec_loss:     {auc_rec:.4f}")
    print(f"  latent_score: {auc_score:.4f}")
    print(f"  energy:       {auc_energy:.4f}")
    


    # ── Per-component stats ──
    for name, arr in [('rec_loss', rec_loss), ('latent_score', latent_score), ('energy', energy)]:
        n_vals = arr[labels == 0]
        a_vals = arr[labels == 1]
        n_mean, a_mean = n_vals.mean(), a_vals.mean()
        n_med, a_med = np.median(n_vals), np.median(a_vals)
        ratio_mean = a_mean / max(n_mean, 1e-12)
        ratio_med = a_med / max(n_med, 1e-12)
        print(f"\n{name}:")
        print(f"  Normal  mean={n_mean:.6f}  median={n_med:.6f}")
        print(f"  Anomaly mean={a_mean:.6f}  median={a_med:.6f}")
        print(f"  Ratio anom/norm  mean={ratio_mean:.2f}x  median={ratio_med:.2f}x")
        
        if ratio_mean < 1.0:
            print(f"  *** WARNING: anomaly values LOWER than normal — signal is INVERTED ***")
    
    print(f"  -rec_loss:    {average_precision_score(labels, -rec_loss):.4f}")
    print(f"  -latent:      {average_precision_score(labels, -latent_score):.4f}")
    print(f"  -energy:      {average_precision_score(labels, -energy):.4f}")

    # ── IQR-based evaluation (only if use_iqr=1) ──
    if args.use_iqr:
        train_energy_list = []
        with torch.no_grad():
            for input_data, _ in train_loader:
                input = input_data.float().to(DEVICE)
                output_dict = model(input)
                output, queries, mem_items = output_dict['out'], output_dict['queries'], output_dict['mem']
                rl = torch.mean(criterion(input, output), dim=-1)
                ls = torch.softmax(gathering_loss(queries, mem_items) / temperature, dim=-1)
                train_energy_list.append((ls * rl).cpu().numpy())
        train_energy_all = np.concatenate(train_energy_list, axis=0).flatten()

        q1 = np.percentile(train_energy_all, 25)
        q3 = np.percentile(train_energy_all, 75)
        iqr = q3 - q1
        thresh_iqr = q3 + 1.5 * iqr
        pred_iqr = (energy > thresh_iqr).astype(int)
        tp = int(((pred_iqr == 1) & (labels == 1)).sum())
        fp = int(((pred_iqr == 1) & (labels == 0)).sum())
        fn = int(((pred_iqr == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"\nIQR threshold: {thresh_iqr:.6f} (Q1={q1:.6f}, Q3={q3:.6f}, IQR={iqr:.6f})")
        print(f"  TP={tp}, FP={fp}, FN={fn}")
        print(f"  Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    suffix = f"_noiqr" if not args.use_iqr else ""

    for idx, (name, arr) in enumerate([('rec_loss', rec_loss), ('latent_score', latent_score), ('energy', energy)]):
        col = idx % 3

        normal_vals = arr[labels == 0]
        anom_vals = arr[labels == 1]

        ax = axes[0, col]
        ax.hist(normal_vals, bins=100, alpha=0.6, label='Normal', color='blue', density=True)
        ax.hist(anom_vals, bins=100, alpha=0.6, label='Anomaly', color='red', density=True)
        ax.set_yscale('log')
        ax.set_title(f'{name} distribution{suffix}')
        ax.legend()

        ax = axes[1, col]
        ax.boxplot([normal_vals[:5000], anom_vals[:5000]], labels=['Normal', 'Anomaly'])
        ax.set_yscale('log')
        ax.set_title(f'{name} box plot{suffix}')

    plt.tight_layout()
    save_path = os.path.join(PROJECT_ROOT, "png", f"energy_diagnosis{suffix}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")

    # ── Conclusion ──
    print(f"\n{'='*60}")
    print(f"  DIAGNOSIS (use_iqr={args.use_iqr}):")
    if auc_rec < 0.1 and auc_score < 0.1:
        print("  Both rec_loss AND latent_score fail to separate anomalies.")
        print("  The model has NOT learned useful representations.")
        print("  Possible fixes:")
        print("    - Increase num_epochs / adjust learning rate")
        print("    - Check if entropy loss is too strong")
        print("    - Try different n_memory (32, 64, 256)")
    elif auc_rec > auc_score + 0.05 and auc_energy < auc_rec:
        print("  rec_loss HAS signal, but latent_score DILUTES it.")
        print("  The multiplicative energy (rec_loss * latent_score) is WORSE")
        print("  than rec_loss alone.")
        print("  ACTION: switch scoring to additive or rec_loss-only.")
        print("    e.g. energy = rec_loss + lambda * latent_score")
        print("    or simply use rec_loss as anomaly score")
    elif auc_score > auc_rec + 0.05 and auc_energy < auc_score:
        print("  latent_score HAS signal, but rec_loss DILUTES it.")
        print("  The model reconstructs anomalies too well (over-generalization).")
        print("  ACTION: increase n_memory or reduce decoder capacity.")
    elif auc_energy >= max(auc_rec, auc_score):
        print("  The multiplicative energy is the BEST or tied-best signal.")
        print("  The scoring formula works correctly.")
        print("  If F1 is still 0, the issue is the THRESHOLD, not the scoring.")
        print("  Adjust anomaly_ratio or switch to IQR-based threshold.")
    else:
        print("  Mixed signal — all components have some discriminative power.")
        print(f"  Best single component: {'rec_loss' if auc_rec >= auc_score else 'latent_score'} "
              f"(AUC-PR={max(auc_rec, auc_score):.4f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
