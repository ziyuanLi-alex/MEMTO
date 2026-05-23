"""
可视化 CUSTOM 数据集异常段的统计特性（修订版）。
- f30/f31 正常段也准静态，不作为方差缩减代表
- 选 f18/f14/f12/f11 作极端低方差代表
- 选 f28-f33 作对照组（异常中方差反而增大）
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
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
w = 0.35  # bar width

# ═══════════════════════════════════════════════════════════════════
# 图 1：variance_comparison.png — 全部特征 std ratio 横向柱状图
# ═══════════════════════════════════════════════════════════════════
normal_std = Xt_n.std(axis=0)
anom_std = Xt_a.std(axis=0)
std_ratio = anom_std / (normal_std + 1e-8)

fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))

# 按 f1-f33 顺序排列
colors = ['#e74c3c' if r < 0.2 else ('#f39c12' if r < 0.6 else ('#95a5a6' if r < 1.1 else '#2ecc71')) for r in std_ratio]
ax1.barh(range(32, -1, -1), std_ratio, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
ax1.axvline(1.0, color='black', ls='--', lw=1, alpha=0.5, label='ratio = 1')
ax1.axvline(0.5, color='red', ls=':', lw=1, alpha=0.3, label='ratio = 0.5')
ax1.set_yticks(range(33))
ax1.set_yticklabels(list(reversed(feature_cols)), fontsize=9)
ax1.set_xlabel('异常段 std / 正常段 std', fontsize=11)
ax1.set_title('全部特征标准差比值（异常 / 正常）', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='lower right')
ax1.grid(axis='x', alpha=0.3)
plt.tight_layout()
out1 = os.path.join(OUT_DIR, 'variance_comparison.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"已保存: {out1}")

# ═══════════════════════════════════════════════════════════════════
# 图 1b：均值偏移图 — |Δmean| / σ_normal
# ═══════════════════════════════════════════════════════════════════
normal_mean = Xt_n.mean(axis=0)
anom_mean = Xt_a.mean(axis=0)
mean_shift = np.abs(anom_mean - normal_mean) / (normal_std + 1e-8)

fig1b, ax1b = plt.subplots(1, 1, figsize=(10, 6))

# 按 f1-f33 顺序排列
ms_colors = ['#e74c3c' if v > 0.5 else ('#f39c12' if v > 0.3 else '#95a5a6') for v in mean_shift]
ax1b.barh(range(32, -1, -1), mean_shift, color=ms_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
ax1b.axvline(1.0, color='black', ls='--', lw=1, alpha=0.5, label='1.0σ')
ax1b.axvline(0.5, color='red', ls=':', lw=1, alpha=0.3, label='0.5σ')
ax1b.set_yticks(range(33))
ax1b.set_yticklabels(list(reversed(feature_cols)), fontsize=9)
ax1b.set_xlabel('|异常 mean − 正常 mean| / 正常 std', fontsize=11)
ax1b.set_title('全部特征均值偏移量（以正常段 σ 为单位）', fontsize=12, fontweight='bold')
ax1b.legend(fontsize=9, loc='lower right')
ax1b.grid(axis='x', alpha=0.3)
plt.tight_layout()
out1b_ms = os.path.join(OUT_DIR, 'mean_shift_comparison.png')
fig1b.savefig(out1b_ms, dpi=150, bbox_inches='tight')
plt.close(fig1b)
print(f"已保存: {out1b_ms}")

# ═══════════════════════════════════════════════════════════════════
# 图 1c：variance_groups.png — 三类特征分组柱状图
# ═══════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 4.5))

# Group A
feats_a = ['f1', 'f2', 'f5', 'f10']
x_a = np.arange(len(feats_a))
for i, f in enumerate(feats_a):
    idx = feature_cols.index(f)
    axes2[0].bar(i - w/2, normal_std[idx], w, color='#4a90d9', alpha=0.8, label='正常' if i == 0 else '')
    axes2[0].bar(i + w/2, anom_std[idx], w, color='#e74c3c', alpha=0.8, label='异常' if i == 0 else '')
axes2[0].set_xticks(x_a)
axes2[0].set_xticklabels(feats_a, fontsize=10)
axes2[0].set_ylabel('标准差', fontsize=10)
axes2[0].set_title('A：均值偏移 + 中等降方差 (ratio ≈ 0.5)', fontsize=11, fontweight='bold')
axes2[0].legend(fontsize=9)
axes2[0].grid(axis='y', alpha=0.3)

# Group B
feats_b = ['f11', 'f12', 'f14', 'f18']
x_b = np.arange(len(feats_b))
for i, f in enumerate(feats_b):
    idx = feature_cols.index(f)
    axes2[1].bar(i - w/2, normal_std[idx], w, color='#4a90d9', alpha=0.8)
    axes2[1].bar(i + w/2, anom_std[idx], w, color='#e74c3c', alpha=0.8)
axes2[1].set_xticks(x_b)
axes2[1].set_xticklabels(feats_b, fontsize=10)
axes2[1].set_ylabel('标准差', fontsize=10)
axes2[1].set_title('B：高动态正常 → 异常冻结 (ratio 0.05-0.21)', fontsize=11, fontweight='bold')
axes2[1].grid(axis='y', alpha=0.3)

# Group C
feats_c = ['f28', 'f29', 'f30', 'f31', 'f32', 'f33']
x_c = np.arange(len(feats_c))
for i, f in enumerate(feats_c):
    idx = feature_cols.index(f)
    axes2[2].bar(i - w/2, normal_std[idx], w, color='#4a90d9', alpha=0.8, label='正常' if i == 0 else '')
    axes2[2].bar(i + w/2, anom_std[idx], w, color='#e74c3c', alpha=0.8, label='异常' if i == 0 else '')
axes2[2].set_xticks(x_c)
axes2[2].set_xticklabels(feats_c, fontsize=10)
axes2[2].set_ylabel('标准差', fontsize=10)
axes2[2].set_title('C：异常中波动反而增大 (ratio ≥ 0.5)', fontsize=11, fontweight='bold')
axes2[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
out1b = os.path.join(OUT_DIR, 'variance_groups.png')
fig2.savefig(out1b, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"已保存: {out1b}")

# ═══════════════════════════════════════════════════════════════════
# 图 2：异常段典型时序（f1/f5/f12/f18，去掉 f30/f31）
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
margin = 60
ws = max(0, s0 - margin)
we = min(len(Xt), e0 + margin)
t = np.arange(ws, we)

for i, feat in enumerate(['f1', 'f5', 'f12', 'f18']):
    idx = feature_cols.index(feat)
    vals = Xt[ws:we, idx]
    axes[i].plot(t, vals, color='#4a90d9', linewidth=0.8, alpha=0.7)
    ai = np.arange(max(ws, s0), min(we, e0 + 1))
    axes[i].fill_betweenx([vals.min(), vals.max()], s0, e0, color='red', alpha=0.1)
    axes[i].set_ylabel(feat, fontsize=11, fontweight='bold', rotation=0, labelpad=30)
    if i == 0:
        axes[i].set_title('异常段附近典型特征时间序列（段 0）', fontsize=12, fontweight='bold')
    axes[i].grid(alpha=0.2)

axes[-1].set_xlabel('测试段索引', fontsize=11)
plt.tight_layout(h_pad=0.5)
out2 = os.path.join(OUT_DIR, 'anomaly_timeseries.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: {out2}")

# ═══════════════════════════════════════════════════════════════════
# 图 3：PCA 分析（扩展版）
# ═══════════════════════════════════════════════════════════════════
Xt_scaled = scaler.transform(Xt)
Xn_scaled = Xt_scaled[normal_mask]
Xa_scaled = Xt_scaled[anom_mask]
pca = PCA(n_components=15).fit(X_train_normal)
Xt_n_pca = pca.transform(Xn_scaled)
Xt_a_pca = pca.transform(Xa_scaled)
explained = pca.explained_variance_ratio_

# 3a: 解释方差比
fig3a, ax3a = plt.subplots(1, 1, figsize=(9, 5))
ax3a.bar(range(1, 16), explained[:15] * 100, color='#4a90d9', alpha=0.8, edgecolor='white')
ax3a.plot(range(1, 16), np.cumsum(explained[:15]) * 100, 'ro-', linewidth=2, markersize=5, label='累积')
for k in [80, 90]:
    ax3a.axhline(y=k, color='gray', ls='--', alpha=0.4)
    ax3a.text(15.3, k, f'{k}%', fontsize=8, color='gray')
ax3a.set_xlabel('主成分', fontsize=11)
ax3a.set_ylabel('解释方差比 (%)', fontsize=11)
ax3a.set_title('PCA 解释方差比（正常数据拟合）', fontsize=12, fontweight='bold')
ax3a.set_xticks(range(1, 16))
ax3a.legend(fontsize=10)
ax3a.grid(axis='y', alpha=0.3)
for i, v in enumerate(explained[:15]):
    ax3a.text(i + 1, v * 100 + 0.3, f'{v*100:.1f}', ha='center', fontsize=6)
plt.tight_layout()
out3a = os.path.join(OUT_DIR, 'pca_explained_variance.png')
plt.savefig(out3a, dpi=150, bbox_inches='tight')
plt.close()

# 3b: 多视角散点 (PC1-PC2, PC1-PC3, PC2-PC3)
fig3b, axes3b = plt.subplots(1, 3, figsize=(18, 5))
pairs = [(0, 1, 'PC1 vs PC2'), (0, 2, 'PC1 vs PC3'), (1, 2, 'PC2 vs PC3')]
for j, (pi, pj, title) in enumerate(pairs):
    axes3b[j].scatter(Xt_n_pca[:, pi], Xt_n_pca[:, pj], s=0.5, alpha=0.06, color='#4a90d9')
    axes3b[j].scatter(Xt_a_pca[:, pi], Xt_a_pca[:, pj], s=8, alpha=0.7, color='#e74c3c',
                      edgecolors='darkred', linewidths=0.5)
    axes3b[j].set_xlabel(f'PC{pi+1} ({explained[pi]*100:.1f}%)', fontsize=10)
    axes3b[j].set_ylabel(f'PC{pj+1} ({explained[pj]*100:.1f}%)', fontsize=10)
    axes3b[j].set_title(title, fontsize=12, fontweight='bold')
    axes3b[j].grid(alpha=0.2)
plt.tight_layout()
out3b = os.path.join(OUT_DIR, 'pca_scatter_pairs.png')
plt.savefig(out3b, dpi=150, bbox_inches='tight')
plt.close()

# 3c: 轨迹对比
fig3c, axes3c = plt.subplots(1, 2, figsize=(16, 5))
X_seg0 = Xt_scaled[s0:e0+1]
pc_seg0 = pca.transform(X_seg0)

axes3c[0].scatter(Xt_n_pca[:, 0], Xt_n_pca[:, 1], s=0.5, alpha=0.06, color='#4a90d9', label='正常云')
axes3c[0].plot(pc_seg0[:, 0], pc_seg0[:, 1], 'o-', color='#e74c3c', linewidth=2, markersize=4, alpha=0.9, label='异常段 0 轨迹')
axes3c[0].scatter(pc_seg0[0, 0], pc_seg0[0, 1], s=100, marker='o', facecolors='none', edgecolors='green', linewidths=2.5, label='起点')
axes3c[0].scatter(pc_seg0[-1, 0], pc_seg0[-1, 1], s=100, marker='s', facecolors='none', edgecolors='purple', linewidths=2.5, label='终点')
axes3c[0].set_xlabel(f'PC1 ({explained[0]*100:.1f}%)', fontsize=11)
axes3c[0].set_ylabel(f'PC2 ({explained[1]*100:.1f}%)', fontsize=11)
axes3c[0].set_title('异常段在 PCA 空间的轨迹', fontsize=12, fontweight='bold')
axes3c[0].legend(fontsize=10, markerscale=3)
axes3c[0].grid(alpha=0.2)

ns, ne = s0 - 30, s0 - 1
X_seg_n = Xt_scaled[ns:ne+1]
pc_seg_n = pca.transform(X_seg_n)
axes3c[1].scatter(Xt_n_pca[:, 0], Xt_n_pca[:, 1], s=0.5, alpha=0.06, color='#4a90d9', label='正常云')
axes3c[1].plot(pc_seg_n[:, 0], pc_seg_n[:, 1], '-', color='#2ecc71', linewidth=1.5, alpha=0.8, label='正常段（异常前 30 点）')
axes3c[1].plot(pc_seg0[:, 0], pc_seg0[:, 1], '-', color='#e74c3c', linewidth=2, alpha=0.9, label='异常段')
axes3c[1].set_xlabel(f'PC1 ({explained[0]*100:.1f}%)', fontsize=11)
axes3c[1].set_ylabel(f'PC2 ({explained[1]*100:.1f}%)', fontsize=11)
axes3c[1].set_title('正常段 vs 异常段轨迹对比', fontsize=12, fontweight='bold')
axes3c[1].legend(fontsize=10)
axes3c[1].grid(alpha=0.2)
plt.tight_layout()
out3c = os.path.join(OUT_DIR, 'pca_trajectory.png')
plt.savefig(out3c, dpi=150, bbox_inches='tight')
plt.close()

# 3d: 各特征 SPE 对比
Xn_rec = pca.inverse_transform(pca.transform(Xn_scaled))
Xa_rec = pca.inverse_transform(pca.transform(Xa_scaled))
res_n = Xn_scaled - Xn_rec
res_a = Xa_scaled - Xa_rec
spe_f_n = np.mean(res_n ** 2, axis=0)
spe_f_a = np.mean(res_a ** 2, axis=0)
spe_ratio_feat = spe_f_a / (spe_f_n + 1e-8)

sig_idx = np.where((spe_ratio_feat < 0.5) | (spe_ratio_feat > 1.5))[0]
sig_feats = [feature_cols[i] for i in sig_idx]
sig_ratios = [spe_ratio_feat[i] for i in sig_idx]
sig_colors = ['#e74c3c' if r < 1 else '#2ecc71' for r in sig_ratios]

fig3d, ax3d = plt.subplots(1, 1, figsize=(10, 5))
x_sf = np.arange(len(sig_feats))
ax3d.bar(x_sf, sig_ratios, color=sig_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
ax3d.axhline(1.0, color='black', ls='--', lw=1, alpha=0.5)
ax3d.set_xticks(x_sf)
ax3d.set_xticklabels(sig_feats, fontsize=9, rotation=45, ha='right')
ax3d.set_ylabel('异常 SPE / 正常 SPE', fontsize=11)
ax3d.set_title('各特征 PCA 重构误差比值（异常 / 正常）', fontsize=12, fontweight='bold')
ax3d.grid(axis='y', alpha=0.3)
plt.tight_layout()
out3d = os.path.join(OUT_DIR, 'pca_feature_spe.png')
plt.savefig(out3d, dpi=150, bbox_inches='tight')
plt.close()

print(f"已保存: {out3a}, {out3b}, {out3c}, {out3d}")

# 打印统计
spe_n = np.mean(res_n ** 2, axis=1)
spe_a = np.mean(res_a ** 2, axis=1)
print(f"\nSPE normal:  mean={spe_n.mean():.6f}, median={np.median(spe_n):.6f}")
print(f"SPE anomaly: mean={spe_a.mean():.6f}, median={np.median(spe_a):.6f}")
print(f"SPE 累积方差: 前10PC={np.sum(explained[:10])*100:.2f}%, 前15PC={np.sum(explained[:15])*100:.2f}%")
