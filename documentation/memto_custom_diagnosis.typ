#import "@preview/grape-suite:4.0.0": subtype

#show: subtype.essay.with(
  title: [MEMTO 在 CUSTOM 数据集上的效果],
  university: [Sustech],
  institute: [Electrical and Electronic Engineering],
  seminar: [Time-Series Anomaly Detection],
  semester: [Spring 2026],
  instructor: [N/A],
  author: [Ziyuan Li],
  date: datetime.today().display("[month repr:long] [day padding:zero], [year repr:full]"),
  show-outline: false,
  colors-primary: rgb("#003f47"),
)

= 数据集特征

== 数据文件与切分

`data/` 目录中主要有三个文件：

- `train.csv`：带有 `y` 标签，用于训练和评估。
- `test_simple.csv`：无标签测试文件。用于提交。
- `test_complex.csv`：无标签测试文件。用于提交。

`train.csv` 的基本统计如下：

#table(
  columns: 2,
  inset: 6pt,
  align: (left, right),
  [项目], [数值],
  [总行数], [137192],
  [特征数], [33],
  [异常点数], [570],
  [全局异常比例], [0.42%],
  [异常段数], [19],
  [每段异常长度], [30 points],
)

测试段包含 `20009` 个正常点和 `570` 个异常点。异常全部位于序列末尾。

== 异常段的统计特性

19 段异常全部为固定长度 30 点。下面从特征的统计学特性展开分析。

=== 方差与均值

#figure(
  image("../png/variance_comparison.png", width: 100%),
  caption: [全部 33 个特征的标准差比值（异常 / 正常）。红色： < 0.2；橙色：0.2~0.6；灰色：0.6~1.1；绿色：> 1.1。],
)

#figure(
  image("../png/mean_shift_comparison.png", width: 100%),
  caption: [全部 33 个特征的均值偏移量（以正常段 σ 为单位）。红色：> 0.5σ；橙色：0.3~0.5σ；灰色：< 0.3σ。],
)

我们直接取出所有 `y=1` 的数据点对其特征进行统计分析，可以观察到，相比于正常区间，大多数异常数据点的特征方差要小于正常段的特征方差，同时 `f1`到`f10` 的均值也有比较明显的移动。

#figure(
  image("../png/anomaly_timeseries.png", width: 100%),
  caption: [异常段附近典型特征的时间序列（段 0，红色阴影区域为异常段）。],
)

通过将数据可视化后，我们可以看到部分feature（f1到f5）在异常区段常常出现人眼可以辨别出来的三角波形 ，而有一些特征(f6到f10)则呈现出阶跃函数特征。f16到f20常常在窗口内出现对称的正脉冲或者负脉冲，有一些feature如f28和f29则不时呈现出类似斜坡函数的时序性特征。这些窗口的特征虽然在单个异常窗下比较明显，但是高度时变，可能此窗口在某几个通道上呈现出特定的三角波形，在下一个窗口这几个通道又会呈现出变化方向相反的类似三角波或者是阶跃函数等。同时，我们有观察到不同的通道有一定的相关性，例如f1到f5五个通道的异常特征通常同步出现，其异常特征的转换也通常同步进行，f6到f10也同理。

=== PCA 结构分析

*正常段分析。* 在标准化后的正常训练数据上拟合 PCA（10 个主成分），累计解释 `74.1%` 的方差。PC1 贡献 `26.1%`，PC2-PC5 各约 `5.6%~5.8%`，之后递减。这说明 33 个特征之间存在较强的线性冗余。

#figure(
  image("../png/pca_normal.png", width: 100%),
  caption: [正常段 PCA 分析。左：解释方差比，前 10 个主成分累计解释 `74.1%`；右：正常段在 PC1-PC2 空间的分布。],
)

*异常段分析与比较。* 我们可以尝试将异常点进行PCA分析之后投影到相同的坐标系上，进行比较。可以发现，红色的数据点与蓝色的数据云重合程度比较高，证明该数据集中的异常可能不适合直接使用PCA进行判断。

#figure(
  image("../png/pca_comparison.png", width: 100%),
  caption: [异常段 PCA 分析。左：异常点（红色）叠加正常云（蓝色）；右：SPE，红色阴影为异常段。],
)

*SPE（Squared Prediction Error）分析。* SPE 衡量的是数据点偏离 PCA 正常子空间的程度。具体而言，将数据点 $x$ 投影到前 $k$ 个主成分上后再反投影回来，得到重构 $hat(x)$，SPE 即为两者之间的均方误差：

$ "SPE"(x) = 1/d sum_(j=1)^d (x_j - hat(x)_j)^2 $

SPE 的物理含义是数据中无法被主成分解释的"残差"部分。如果一个数据点的特征间协方差关系与正常模式一致，它主要落在主成分方向上，残差小、SPE 低；反之，如果协方差结构被破坏，更多方差会泄露到残差子空间中，SPE 升高。

但是在当前的数据集中，这个方法不太适用：首先，从直接观察来看，经过滑动滤波的异常段SPE和正常段的SPE并没有显著的区别。正常情况下我们期望异常段的重构误差更高，因为异常应该偏离正常的子空间。但这里的异常段存在内部规律且方差降低，导致它们在残差子空间中的投影变化不太显著，所以简单的重构方法对这个数据集不是非常适用。于是我们就在想，现在主流的生成式深度学习方法是否可以更有效的对数据集中的异常进行识别？

= MEMTO 方法

== 整体架构

MEMTO（Memory-guided Transformer for Multivariate Time Series Anomaly Detection）是一个发表于2023年NeurIPS的时序数据异常检测模型。它的核心思路是：*模型只在"正常数据"上训练，在异常数据部分重建能力较差，从而把异常检测转化为重建质量判断。*

具体来说，给定一个长度为 $L$ 的窗口（含 $n$ 个传感器通道），数据依次经过三步：

1. *Transformer Encoder*：把原始数据编码为高维表示 $q^s$。Encoder 是标准的自注意力结构，负责捕捉多变量之间的时序依赖关系。在我们的实现中，$d_"model" = 512$，即每个时间步被映射到一个 512 维的向量。

2. *Gated Memory Module*：这是 MEMTO 的核心创新。模块中预先存储了 $M$ 个"正常原型"向量（记忆项），每个代表一种典型的正常模式。正常数据能很好地匹配到某个记忆项，异常数据则难以匹配。

3. *Weak Decoder*：一个非常简单的两层全连接网络，把 $hat(q)^s$ 映射回原始 $n$ 维空间，得到重建 $hat(X)^s$。这个Decoder结构较为简单，是为了抑制模型的重建能力，防止模型"过泛化"。


= 实验结果

== MEMTO 性能

使用作者公开的代码，在数据集上训练 MEMTO，测试结果如下：

#table(
  columns: 2,
  inset: 6pt,
  align: (left, right),
  [Metric], [Value],
  [Accuracy], [0.9698],
  [Precision], [0.4286],
  [Recall], [0.2632],
  [F-score], [0.3261],
  [AUC-PR], [0.0279],
  [Range-F], [0.0448],
)

测试段异常比例约为 `2.8%`，而 MEMTO 的 AUC-PR 只有 `0.0279`，几乎等于随机排序。

== 组件分析

将 MEMTO 的 energy 分解为三部分后，各组件的 AUC-PR 均接近随机：

#table(
  columns: 2,
  inset: 6pt,
  align: (left, right),
  [Component], [AUC-PR],
  [`rec_loss`], [0.0349],
  [`latent_score`], [0.0277],
  [`energy`], [0.0279],
)

= 讨论

== 模型假设与异常类型不匹配

MEMTO 采用 reconstruction-based objective，其前提是异常数据比正常数据更难重构。但是本模型中的异常段具有较为明显的模式且方差较小，模型有可能反而更加容易对这些段落进行重构。同时，不同通道内数据的异常模式会随时间产生变化，这也有可能干扰了该模型对异常段落的识别。

= 附录：异常模式时序示意

以下展示四个典型异常段的多通道时序图，红色阴影区域为异常段（30 点）。

== 异常段 6（索引 19595–19624）

#figure(
  image("../png/anomaly_pattern_seg6.png", width: 100%),
  caption: [异常段 6 的时序模式。f1-f5 在异常段呈现负的阶跃函数；f6-f10 呈现负的三角波；f11-f15 和 f16-f21 除了一个通道有负向阶跃函数，其他通道多为静态接近0的数值；f29出现斜坡。这些特定函数模式恰好具有较低方差和均值上的偏移。],
)

== 异常段 2（索引 17137–17166）

#figure(
  image("../png/anomaly_pattern_seg2.png", width: 100%),
  caption: [异常段 2 的时序模式。与段 6 类似。],
)

== 异常段 11（索引 21601–21630）

#figure(
  image("../png/anomaly_pattern_seg11.png", width: 100%),
  caption: [异常段 11 的时序模式。三角波来到了 f1-f5 且方向相反，f6-f10 的阶跃幅度和方向也有所不同f11-f15出现了被2个负向冲激函数包裹的平台。],
)

== 异常段 15（索引 24792–24821）

#figure(
  image("../png/anomaly_pattern_seg15.png", width: 100%),
  caption: [异常段 15 的时序模式。四类基本函数模式（三角波、阶跃、脉冲、斜坡）在不同通道交替出现。],
)


