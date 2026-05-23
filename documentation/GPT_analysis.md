
• 已阅读 documentation/memto_custom_diagnosis.pdf。这份报告主要分析了 MEMTO 在 CUSTOM 时序异常检测数据集上的效果，结论是：MEMTO 在该数据集上表现接近随机，核心原因是数
  据集异常类型与 MEMTO 的重建式异常检测假设不匹配。

  核心内容总结

  1. 数据集特征
     train.csv 共 137,192 行、33 个特征、570 个异常点，异常比例仅 0.42%。异常被组织成 19 段，每段固定 30 个点。测试段包含 20,009 个正常点和 570 个异常点，异常全部位于
     序列末尾。
  2. 异常模式特征
     异常段不是噪声型或明显离群型异常，而是具有规律结构：
      - 多数异常点的特征方差低于正常段。
      - f1 到 f10 存在较明显均值偏移。
      - f1-f5 常出现三角波形。
      - f6-f10 常出现阶跃函数。
      - f16-f20 常出现正/负脉冲。
      - f28/f29 有时出现斜坡形态。
      - 异常模式在不同异常窗口之间高度变化，但通道组内部有同步相关性。
  3. PCA 分析
     正常数据的前 10 个 PCA 主成分累计解释 74.1% 方差，说明 33 个特征之间有较强线性冗余。但异常点投影到 PC1-PC2 空间后与正常数据云高度重合，说明 PCA 难以直接区分这些异
     常。

     SPE 重构误差也不明显，因为异常段虽然有模式变化，但方差更低、结构较规则，不一定偏离正常子空间。
  4. MEMTO 方法简介
     MEMTO 是 2023 NeurIPS 的 Memory-guided Transformer 方法。它只用正常数据训练，通过重建误差判断异常。结构包括：
      - Transformer Encoder：编码多变量时序依赖。
      - Gated Memory Module：存储正常模式原型。
      - Weak Decoder：限制重建能力，避免过度泛化。
  5. 实验结果
     在 CUSTOM 数据集上训练和测试 MEMTO 后，结果较差：
      - Accuracy: 0.9698
      - Precision: 0.4286
      - Recall: 0.2632
      - F-score: 0.3261
      - AUC-PR: 0.0279
      - Range-F: 0.0448

     报告指出测试异常比例约 2.8%，而 MEMTO 的 AUC-PR 为 0.0279，几乎等于随机排序。
  6. 组件分析
     将 MEMTO 的 energy 拆解后，各部分也都接近随机：
      - rec_loss: 0.0349
      - latent_score: 0.0277
      - energy: 0.0279
  7. 主要结论
     MEMTO 失效的根本原因是它依赖一个假设：异常数据应该比正常数据更难重建。但 CUSTOM 数据集中的异常段具有明显、规则、低方差的函数模式，可能反而更容易被模型重建。因此，
     重建式深度模型并不能有效识别这类异常。
  8. 附录
     报告展示了异常段 2、6、11、15 的多通道时序图，说明三角波、阶跃、脉冲、斜坡等模式会在不同通道间交替出现，进一步支持“异常有规则但高度时变”的判断。

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


› 针对这样的数据，你比较推荐使用什么架构进行异常检测？


• 我会优先推荐 判别式的时序分割架构，而不是 MEMTO 这类 reconstruction-based 模型。

  原因很直接：这批异常不是“难以重建”的异常，而是低方差、有规则形状、通道组同步变化的局部模式。所以模型应该学“什么形状像异常”，而不是学“什么东西重建不好”。

  首选架构：1D TCN / InceptionTime / U-Net 式时序分割模型

  推荐形式：

  input:  window [T, 33]
          + delta features
          + rolling mean/std/slope features

  backbone:
          multi-scale 1D CNN / TCN blocks
          residual connections
          dilated convolutions
          channel attention or grouped channel mixing

  head:
          per-timestep binary segmentation
          output [T, 1] anomaly probability

  比较适合的具体结构：

  Multi-scale Conv1D kernels: 3, 5, 9, 15, 31
  + Dilated TCN blocks
  + Squeeze-and-Excitation / channel attention
  + Residual connections
  + Binary segmentation head

  训练目标建议用：

  BCE + Dice loss
  或 Focal loss + Dice loss

  因为异常点极少，普通 BCE 很容易被正常点淹没。

  为什么这个架构更适合

  这类数据的异常有明显的局部形状：

  - 三角波
  - 阶跃
  - 脉冲
  - 斜坡
  - 通道组同步变化
  - 异常段固定约 30 点

  CNN/TCN 对这些局部 pattern 很敏感，尤其是多尺度卷积可以同时捕捉短脉冲、30 点异常段、较长趋势变化。相比 Transformer，它对小数据和强局部模式通常更稳，也更不容易过拟
  合。

  我会避免的方向

  不推荐继续主打：

  AutoEncoder / Transformer reconstruction / MEMTO / PCA-SPE

  因为它们默认“异常难以重建”。但这里的异常反而可能更平滑、更规则、更容易重建。

  非常强的 baseline：特征工程 + LightGBM

  如果目标是效果而不是纯深度学习，我会同时做一个窗口级模型：

  window length: 30 / 60 / 90
  features:
    mean shift
    std ratio
    slope
    first/second derivative energy
    max-min
    channel group correlation
    template matching score: step / triangle / pulse / ramp
  model:
    LightGBM / CatBoost

  这个数据的异常结构很“函数模式化”，所以这种方法可能非常强，而且可解释性比深度模型好。

  最终推荐方案

  如果是比赛/追求分数：

  LightGBM 特征模型 + TCN 分割模型 ensemble

  如果是论文/模型方法：

  Multi-scale TCN / InceptionTime-style temporal segmentation network

  如果只能选一个，我会选：

  多尺度 TCN + channel attention + Dice/Focal loss 的逐点异常分割模型

  它和这个数据的异常机制最匹配。


› 那么类似CNN的架构呢？


• 类似 CNN 的架构我会推荐 1D 多尺度卷积时序分割网络，比普通 CNN 分类器更合适。因为你的异常是连续 30 点一段，目标应该是输出每个时间点的异常概率，而不是只判断一个窗口是
  否异常。

  一个比较合适的结构是：

  Input: [batch, time, 33]

  Stem:
    Conv1D(33 -> 64, kernel=3)
    BatchNorm
    GELU

  Multi-scale blocks:
    branch 1: Conv1D(k=3)
    branch 2: Conv1D(k=7)
    branch 3: Conv1D(k=15)
    branch 4: Conv1D(k=31)
    concat
    1x1 Conv
    residual connection

  Temporal context:
    dilated Conv1D blocks
    dilation = 1, 2, 4, 8
    residual connection

  Channel attention:
    Squeeze-and-Excitation
    or lightweight channel attention

  Head:
    Conv1D(64 -> 32, k=3)
    Conv1D(32 -> 1, k=1)
    Sigmoid

  Output: [batch, time]

  这个结构专门适合你报告里描述的异常模式：

  - k=3/7 捕捉脉冲、短阶跃边缘；
  - k=15/31 捕捉 30 点左右的三角波、斜坡、平台；
  - dilated conv 扩大上下文，帮助判断“这一段是不是异常段”；
  - channel attention 捕捉 f1-f5、f6-f10 这种通道组同步异常。

  我不建议只做普通 2D CNN 图像化

  比如把 [time, feature] 当成图片喂 2D CNN，这可以试，但不是首选。因为 feature 维度上的相邻关系是人为编号，不一定像图像像素那样有空间连续性。f1-f5、f6-f10 有组相关性，
  但不是完整二维平移不变结构。

  更稳的做法是：

  时间维度用 Conv1D
  通道维度用 1x1 Conv / channel attention / grouped conv

  如果你想让模型显式利用通道组，可以用 grouped branch：

  Group A: f1-f5
  Group B: f6-f10
  Group C: f11-f15
  Group D: f16-f21
  Group E: f22-f33

  每组单独 Conv1D 提取局部模式
  再 concat 后做 1x1 Conv 融合

  一个强 baseline CNN

  InceptionTime-style 1D CNN

  也就是多个不同 kernel size 的 Conv1D 并联：

  Conv1D k=3
  Conv1D k=5
  Conv1D k=9
  Conv1D k=17
  Conv1D k=31
  concat -> 1x1 Conv -> residual

  对三角波、阶跃、脉冲、斜坡这种多尺度形状很合适。

  训练建议

  窗口长度不要只取 30，建议：

  window = 96 或 128
  stride = 1~8
  label = 每个 timestep 的 y

  loss 用：

  Focal loss + Dice loss

  或者：

  weighted BCE + Dice loss

  因为异常比例太低。

  最终我会从这个版本开始：

  Multi-scale 1D CNN / InceptionTime
  + Dilated Conv
  + Channel Attention
  + per-timestep segmentation head

  这是一个和你这个 CUSTOM 数据最匹配的 CNN 系方案。


› CNN 和 你提出的这个LightGBM有什么区别


• 核心区别是：CNN 自动从原始序列里学形状模式，LightGBM 依赖你先把形状总结成表格特征。

  ┌──────────────┬────────────────────────────────────────┬────────────────────────────────────────────────┐
  │ 维度         │ CNN / TCN                              │ LightGBM                                       │
  ├──────────────┼────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ 输入         │ 原始窗口，如 [T, 33]                   │ 人工统计特征表                                 │
  │ 学到什么     │ 三角波、阶跃、脉冲、斜坡等局部时序形状 │ 均值、方差、斜率、差分能量、相关性等已定义特征 │
  │ 特征工程     │ 少一些                                 │ 很重要                                         │
  │ 数据需求     │ 通常更多                               │ 小数据上更稳                                   │
  │ 可解释性     │ 较弱                                   │ 较强                                           │
  │ 对异常比例低 │ 需要 loss/采样处理                     │ 类别权重、采样、阈值调节较方便                 │
  │ 对时序位置   │ 天然建模                               │ 需要滑窗特征表达                               │
  │ 泛化         │ 学得好时更强，但也更容易过拟合         │ 更保守，通常 baseline 很强                     │
  │ 输出         │ 逐点异常概率或窗口异常概率             │ 通常是点/窗口的异常概率                        │
  └──────────────┴────────────────────────────────────────┴────────────────────────────────────────────────┘

  放在你这个数据上：

  CNN 会直接看类似这样的输入：

  过去 96 个时间点 x 33 个特征

  然后自己学习：

  f1-f5 出现同步三角波
  f6-f10 出现同步阶跃
  f16-f20 出现脉冲
  f28/f29 出现斜坡

  LightGBM 不会直接“看见”这些波形。你需要先把窗口变成特征，比如：

  f1-f5 的均值变化
  f1-f5 的标准差下降比例
  f6-f10 的前后均值差
  f16-f20 的最大脉冲幅度
  f28/f29 的线性斜率
  一阶差分能量
  二阶差分能量
  通道组相关性
  三角波模板匹配分数
  阶跃模板匹配分数

  然后 LightGBM 学这些特征和异常标签之间的关系。

  简单说：

  CNN = 让模型自己从波形里学检测规则
  LightGBM = 你把波形压缩成指标，让模型组合这些指标

  我会这样选：

  - 数据少、异常样本少、想快速拿稳定结果：LightGBM
  - 异常模式复杂、希望模型自动学多通道时序形状：CNN/TCN
  - 实战追求最好效果：二者 ensemble

  对你这个数据，我的判断是：LightGBM 可能更快得到强 baseline，CNN 更适合作为上限探索。