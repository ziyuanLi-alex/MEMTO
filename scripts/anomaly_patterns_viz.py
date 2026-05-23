"""
生成 Appendix 用的异常模式时序图：选取几个典型异常段，展示不同特征组的异常模式。
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'png')
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
feature_cols = [c for c in df.columns if c != 'y']
X = np.nan_to_num(df[feature_cols].values.astype(np.float32))
y = df['y'].values

split = int(len(X) * 0.8)
X_test, y_test = X[split:], y[split:]

anom_indices = np.where(y_test == 1)[0]
breaks = np.where(np.diff(anom_indices) > 1)[0]
seg_starts = np.concatenate([[anom_indices[0]], anom_indices[breaks + 1]])
seg_ends = np.concatenate([anom_indices[breaks], [anom_indices[-1]]])


def plot_anomaly_pattern(seg_idx, feature_groups, title, filename, window_size=180):
    """
    seg_idx: 异常段索引
    feature_groups: dict of {group_name: [feat_names]}
    """
    s = seg_starts[seg_idx]
    e = seg_ends[seg_idx]
    ws = max(0, s - window_size // 2)
    we = min(len(X_test), e + window_size // 2)
    t = np.arange(ws, we)

    n_subplots = len(feature_groups)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 2.2 * n_subplots), sharex=True)
    if n_subplots == 1:
        axes = [axes]

    ai = 0  # axes index
    for gname, feats in feature_groups.items():
        for feat in feats:
            idx = feature_cols.index(feat)
            vals = X_test[ws:we, idx]
            axes[ai].plot(t, vals, linewidth=0.6, alpha=0.7, label=feat)
            ai_start = max(ws, s)
            ai_end = min(we, e + 1)
            if ai_start < ai_end:
                axes[ai].plot(np.arange(ai_start, ai_end),
                              X_test[ai_start:ai_end, idx],
                              linewidth=1.2, alpha=0.9)
        vmin = min(X_test[ws:we, feature_cols.index(f)].min() for f in feats)
        vmax = max(X_test[ws:we, feature_cols.index(f)].max() for f in feats)
        if vmax > vmin:
            axes[ai].fill_betweenx([vmin, vmax], s, e + 1, color='red', alpha=0.06)
        axes[ai].axvline(s, color='red', ls='--', lw=0.8, alpha=0.4)
        axes[ai].axvline(e + 1, color='red', ls='--', lw=0.8, alpha=0.4)
        axes[ai].set_ylabel(gname, fontsize=10, fontweight='bold', rotation=0, labelpad=25)
        axes[ai].legend(fontsize=7, loc='upper right', ncol=min(len(feats), 6))
        axes[ai].grid(alpha=0.15)
        ai += 1

    axes[-1].set_xlabel('测试段索引', fontsize=10)
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(h_pad=0.3)
    out = os.path.join(OUT_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {out}")


# ── 图 1：异常段 6 ──────────────────────
plot_anomaly_pattern(
    seg_idx=6,
    feature_groups={
        'f1-f5': ['f1', 'f2', 'f3', 'f4', 'f5'],
        'f6-f10': ['f6', 'f7', 'f8', 'f9', 'f10'],
        'f11-f15': ['f11', 'f12', 'f13', 'f14', 'f15'],
        'f16-f21': ['f16', 'f17', 'f18', 'f19', 'f20', 'f21'],
        'f28-f33': ['f28', 'f29', 'f30', 'f31', 'f32', 'f33'],
    },
    title='异常段 6（索引 19595–19624）',
    filename='anomaly_pattern_seg6.png',
    window_size=180,
)

# ── 图 2：异常段 2 ──────────────────────
plot_anomaly_pattern(
    seg_idx=2,
    feature_groups={
        'f1-f5': ['f1', 'f2', 'f3', 'f4', 'f5'],
        'f6-f10': ['f6', 'f7', 'f8', 'f9', 'f10'],
        'f11-f15': ['f11', 'f12', 'f13', 'f14', 'f15'],
        'f16-f21': ['f16', 'f17', 'f18', 'f19', 'f20', 'f21'],
        'f28-f33': ['f28', 'f29', 'f30', 'f31', 'f32', 'f33'],
    },
    title='异常段 2（索引 17137–17166）',
    filename='anomaly_pattern_seg2.png',
    window_size=180,
)

# ── 图 3：异常段 11 ──────────────────
plot_anomaly_pattern(
    seg_idx=11,
    feature_groups={
        'f1-f5': ['f1', 'f2', 'f3', 'f4', 'f5'],
        'f6-f10': ['f6', 'f7', 'f8', 'f9', 'f10'],
        'f11-f15': ['f11', 'f12', 'f13', 'f14', 'f15'],
        'f16-f21': ['f16', 'f17', 'f18', 'f19', 'f20', 'f21'],
        'f28-f33': ['f28', 'f29', 'f30', 'f31', 'f32', 'f33'],
    },
    title='异常段 11（索引 21601–21630）',
    filename='anomaly_pattern_seg11.png',
    window_size=180,
)

# ── 图 4：异常段 15 ─────────────────
plot_anomaly_pattern(
    seg_idx=15,
    feature_groups={
        'f1-f5': ['f1', 'f2', 'f3', 'f4', 'f5'],
        'f6-f10': ['f6', 'f7', 'f8', 'f9', 'f10'],
        'f11-f15': ['f11', 'f12', 'f13', 'f14', 'f15'],
        'f16-f21': ['f16', 'f17', 'f18', 'f19', 'f20', 'f21'],
        'f28-f33': ['f28', 'f29', 'f30', 'f31', 'f32', 'f33'],
    },
    title='异常段 15（索引 24792–24821）',
    filename='anomaly_pattern_seg15.png',
    window_size=180,
)
