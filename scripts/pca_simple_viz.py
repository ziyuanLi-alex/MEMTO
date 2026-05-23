"""
简化的 PCA 可视化：正常段分析 → 异常段分析 → 比较
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
X_train_normal = X[:split][y[:split] == 0]
scaler = StandardScaler().fit(X_train_normal)

normal_mask = y_test == 0
anom_mask = y_test == 1
Xt = X_test
Xt_n = Xt[normal_mask]
Xt_a = Xt[anom_mask]

# 异常段定位
anom_indices = np.where(y_test == 1)[0]
breaks = np.where(np.diff(anom_indices) > 1)[0]
seg_starts = np.concatenate([[anom_indices[0]], anom_indices[breaks + 1]])
seg_ends = np.concatenate([anom_indices[breaks], [anom_indices[-1]]])
s0, e0 = seg_starts[0], seg_ends[0]

# PCA 拟合（仅正常训练数据）
Xt_scaled = scaler.transform(Xt)
Xn_scaled = Xt_scaled[normal_mask]
Xa_scaled = Xt_scaled[anom_mask]
pca = PCA(n_components=10).fit(X_train_normal)
Xt_n_pca = pca.transform(Xn_scaled)
Xt_a_pca = pca.transform(Xa_scaled)
explained = pca.explained_variance_ratio_

# ═══════════════════════════════════════════════════════════════
# 图 1：正常段 PCA 分析
# ═══════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 2, figsize=(16, 5))

# (a) 解释方差比
axes1[0].bar(range(1, 11), explained * 100, color='#4a90d9', alpha=0.8, edgecolor='white')
axes1[0].plot(range(1, 11), np.cumsum(explained) * 100, 'ro-', linewidth=2, markersize=5, label='累积')
axes1[0].axhline(y=80, color='gray', ls='--', alpha=0.4)
axes1[0].text(10.3, 80, '80%', fontsize=8, color='gray')
axes1[0].set_xlabel('主成分', fontsize=11)
axes1[0].set_ylabel('解释方差比 (%)', fontsize=11)
axes1[0].set_title('正常段 PCA 解释方差比', fontsize=12, fontweight='bold')
axes1[0].set_xticks(range(1, 11))
axes1[0].legend(fontsize=10)
axes1[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(explained):
    axes1[0].text(i + 1, v * 100 + 0.5, f'{v*100:.1f}%', ha='center', fontsize=8)

# (b) PC1-PC2 散点（仅正常点）
axes1[1].scatter(Xt_n_pca[:, 0], Xt_n_pca[:, 1], s=0.5, alpha=0.15, color='#4a90d9')
axes1[1].set_xlabel(f'PC1 ({explained[0]*100:.1f}%)', fontsize=11)
axes1[1].set_ylabel(f'PC2 ({explained[1]*100:.1f}%)', fontsize=11)
axes1[1].set_title('正常段在 PC1-PC2 空间的分布', fontsize=12, fontweight='bold')
axes1[1].grid(alpha=0.2)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, 'pca_normal.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"已保存: {out1}")

# ═══════════════════════════════════════════════════════════════
# 图 2：异常段 SPE 局部窗口（典型段 0，窗口 = 210）
# ═══════════════════════════════════════════════════════════════
Xt_all_rec = pca.inverse_transform(pca.transform(Xt_scaled))
spe_all = np.mean((Xt_scaled - Xt_all_rec) ** 2, axis=1)

s0, e0 = seg_starts[0], seg_ends[0]  # 典型异常段 0
ws = max(0, s0 - 100)  # 窗口起始
we = min(len(Xt_scaled), e0 + 100)  # 窗口结束
window_size = we - ws
t_win = np.arange(ws, we)

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))

# (a) 异常点在 PC1-PC2 的分布（叠加正常云背景）
axes2[0].scatter(Xt_n_pca[:, 0], Xt_n_pca[:, 1], s=0.3, alpha=0.06, color='#4a90d9', label='正常点')
axes2[0].scatter(Xt_a_pca[:, 0], Xt_a_pca[:, 1], s=6, alpha=0.6, color='#e74c3c',
                 label='异常点', edgecolors='darkred', linewidths=0.5)
axes2[0].set_xlabel(f'PC1 ({explained[0]*100:.1f}%)', fontsize=10)
axes2[0].set_ylabel(f'PC2 ({explained[1]*100:.1f}%)', fontsize=10)
axes2[0].set_title('异常点分布 vs 正常云', fontsize=12, fontweight='bold')
axes2[0].legend(fontsize=9, markerscale=5)
axes2[0].grid(alpha=0.2)

# (b) SPE 局部窗口时序 — 段 0，窗口 210
spe_win = spe_all[ws:we]
axes2[1].plot(t_win, spe_win, color='#4a90d9', linewidth=1.0, alpha=0.8, label='逐点 SPE')
axes2[1].plot(t_win, np.convolve(spe_win, np.ones(10)/10, mode='same'),
             color='#2c3e50', linewidth=2.0, alpha=0.9, label='滑动平均 (10)')

# 高亮异常段
axes2[1].fill_betweenx([spe_win.min(), spe_win.max()], s0, e0 + 1, color='red', alpha=0.12)
axes2[1].axvline(s0, color='red', ls='--', lw=1.2, alpha=0.6, label=f'异常段 [{s0}, {e0}]')
axes2[1].axvline(e0 + 1, color='red', ls='--', lw=1.2, alpha=0.6)

axes2[1].set_xlabel('测试段索引', fontsize=11)
axes2[1].set_ylabel('SPE', fontsize=11)
axes2[1].set_title(f'SPE 局部窗口（异常段 0，{window_size} 点）', fontsize=12, fontweight='bold')
axes2[1].legend(fontsize=9, loc='upper right')
axes2[1].grid(alpha=0.2)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, 'pca_comparison.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"已保存: {out2}")

# 打印
print(f"\nPCA 解释方差: 前10PC={np.sum(explained)*100:.2f}%")
print(f"SPE normal:  mean={spe_n.mean():.6f}, median={np.median(spe_n):.6f}")
print(f"SPE anomaly: mean={spe_a.mean():.6f}, median={np.median(spe_a):.6f}")
