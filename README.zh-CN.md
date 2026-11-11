# msrspatial

**中文** | [English](README.md)

**在 Moran 空间零模型下检验脑图之间的空间关联。**

`msrspatial` 用于检验两组脑区级图谱之间是否存在空间关联。它使用的零模型**保留空间自相关**
（Moran Spectral Randomization，MSR），而不是随机打乱数值。它是影像转录组流程中可复用的核心：
把一组耦合图与一组分子图（受体密度、基因表达评分）配对，输出效应量、空间 p 值与 FDR 校正后的 q 值。

## 为什么必须用空间零模型

两张平滑的皮层图，仅凭偶然就可能相关到 |r| > 0.5。朴素置换会破坏这种平滑性，因而给出的 p 值**偏松**
（anti-conservative）。MSR 生成与观测图具有**相同空间自相关结构**的替代图，于是检验问的是正确的问题：

> 这个关联，是否强于"同样平滑的随机图"所能产生的关联？

## 安装

```bash
pip install msrspatial
# 开发模式
pip install -e ".[dev,plot,enrich]"
```

要求 Python ≥ 3.9，依赖 numpy、pandas、scipy。

## 快速开始（Python）

```python
from msrspatial import spatial_association, load_matrix

maps = load_matrix("coupling_maps.csv", id_column="region")   # (脑区数, 图谱数)
features = load_matrix("receptors.csv", id_column="region")   # (脑区数, 特征数)
distance = load_matrix("distance.csv")                        # (脑区数, 脑区数)

result = spatial_association(
    maps, features, distance,
    map_names=["MFPT", "Reliance"],
    feature_names=["D1", "DAT", "MOR"],
    knn=15, n_rep=1000, seed=2026,
)

result.table            # map, feature, r, p_spin, q_fdr, significant
result.significant()    # q_fdr < 0.05
result.diagnostics      # 每个图谱的零模型质控
```

## 快速开始（命令行）

```bash
msrspatial run \
  --maps coupling_maps.csv --features receptors.csv --distance distance.csv \
  --id-column region --knn 15 --n-perm 1000 --seed 2026 \
  --out results/

msrspatial demo --out demo_results      # 内置合成数据，无需下载
```

输出 `spatial_association.csv` 与 `diagnostics.json`。

## 输入格式

| 输入 | 形状 | 支持格式 |
|---|---|---|
| `maps` | 脑区 × 图谱 | `.npy`、CSV/TSV（有无表头均可） |
| `features` | 脑区 × 特征 | 带 id 列的 CSV/TSV |
| `distance` | 脑区 × 脑区 | `.npy`、CSV/TSV（有无表头均可） |

三个输入的行顺序**必须一致**。`validate_alignment` 会在以下情况直接报错：行数不匹配、
含 NaN/Inf、距离矩阵不对称、对角线非零。**行序错位是空间分析不可复现最常见的原因**，
因此这里选择"报错"而不是"静默继续"。

## 零模型质控

`result.diagnostics["per_map_diagnostics"]` 对每个图谱报告：

| 指标 | 期望 | 含义 |
|---|---|---|
| `mean_ratio` | ≈ 1 | 替代图的边际均值是否保留 |
| `sd_ratio` | ≈ 1 | 替代图的边际标准差是否保留 |
| `morans_i_ratio` | ≈ 1 | **空间自相关是否保留**（MSR 的定义性属性） |
| `mean_corr` | ≈ 0 | 与观测图相关的**有符号均值** |
| `sd_corr` | 可能很大 | 单个替代图可能接近 +x 或 −x；MSR 随机化的是相位，不是幅度 |

若 `morans_i_ratio` 明显偏离 1，说明零模型不再是"空间零模型"，此时 p 值不可解释。

> **注意**：不要用"替代图与观测图是否去相关"来判断零模型好坏。MSR 的替代图**故意保留**
> 空间自相关，因此单次相关可能很高（±0.9），只有**均值**才应接近 0。

## 方法

1. 空间权重：逆距离 k 近邻图，对称化。
2. 对双重中心化后的权重矩阵做特征分解，得到 Moran 特征向量（MEM）。
3. 对每个观测图，投影到 MEM 基底，随机翻转各系数符号，重构得到 `n_rep` 个替代图。
4. 用向量化的 Pearson / Spearman 计算替代图与特征的关联。
5. 双侧 p 值，采用保守的 `(1 + 超过数) / (K + 1)` 校正。
6. Benjamini–Hochberg FDR：默认在每个图谱内校正（`per_map`），也可全局校正（`global`）。

参考文献：Wagner & Dray (2015) *Methods in Ecology and Evolution* 6:1169–1179；
Vos de Wael et al. (2020) *NeuroImage* 221:117112（brainspace）。

## 适用范围与限制

- 面向**脑区级**图谱（如 HCP-MMP 360）。顶点级数据需先分区。
- 要求距离矩阵**对称**且对角线为零。
- `n_rep=1000` 的 p 值分辨率为 0.001；正式发表阈值建议更高。
- FDR 的检验族必须在看结果**之前**声明。

## 测试

```bash
pytest -q                     # 若已安装 pytest
python examples/selftest.py   # 无需 pytest 的自检
```

## 引用

使用本软件请引用描述它的论文（见 `CITATION.cff`）以及上述方法学文献。

## 许可

BSD-3-Clause。
