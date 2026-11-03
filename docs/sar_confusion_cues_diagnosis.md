# SAR-AIRcraft：型号混淆线索诊断（CPU）

日期：2026-11-02。前置结论见 [主模块研判 §2](sar_rem_pyramid_experiment.md)，结果台账见 [结果总表](sar_results_ledger.md)。

## 为什么做

test 上 baseline 的 AP50 型号缺口为 16.1，主要来自窄体机：Boeing737→Boeing787 占 test 737 的 19.8%，
其后是 A220→other、A220→ARJ21、ARJ21→Boeing737。源图尺寸先验已排除（737 与 787 的源图尺寸几乎相同），
改分类头的方案也都没有通过 test。设计第二模块之前，需要先弄清楚：模型缺的是哪种信息。

工具 `tools/analyze_confusion_cues.py` 只读已有诊断目录（`compare_rem_102e.py` 的输出）中的预测和原图，
只用 CPU，不推理、不训练，不改动已有诊断文件。输出在 `DIAG/confusion_cues/`。

## 分析内容与判读

焦点为 baseline 混淆最多的 4 对。“定位”的定义与已有诊断相同：任意类别预测，score≥0.3，IoU≥0.5，不做一对一匹配。

| 节 | 内容 | 若成立 → 第二模块方向 |
|---|---|---|
| 1 分数排序 | 定位后真类在同一框各类分数中的名次；判错时真类排第 2 的比例、与第 1 的分数差 | 判错多为真类第 2 且分数差小 → 信息已有、只是排序偏，做分数侧校准（CLCR-Lite 一类）；真类常缺失 → 特征里没有区分信息 |
| 2 几何线索 | 每对 a/b 用源图尺寸、640 尺寸、**同图相对尺寸**（本机 max(w,h) / 同图其他飞机中位数，GT 与可推理的预测两个版本）、长宽比做 AUC；并比较 a 正确、a→b、b 正确三组的相对尺寸中位数 | 相对尺寸可分性高（≳0.8）而源图尺寸低，且 a→b 的相对尺寸更接近 a → 加入同图目标间的相对尺度上下文（query 之间的尺度关系） |
| 3 图像集中度 | 每类判错在图像间的分布：离散度（1 = 独立分布，>1 = 成簇）、一半错误所在图数、来自“≥80% 判错”图的错误比例 | 高度成簇 → 成像条件/场景或标注一致性问题，先看裁剪图，模块未必能解决 |
| 4 拥挤程度 | 按最近邻距离 / 自身尺寸、同图 GT 数分组的定位率与定位后型号正确率 | 密集停机坪上明显更差 → 目标间干扰，考虑邻域抑制/分离 |
| 5 裁剪图 | 每对一张 PNG：a 判成 b（红）、a 正确（绿）、b 正确（蓝），固定 176 源图像素窗口不缩放，保留真实尺寸；每组随机 16 个（seed 0） | 人工看外观差别：机身长宽比、后掠、发动机/尾翼散射点 |

注意：

- 预测导出为 query×class 展平后的 top-300，某些类别的分数可能不在导出中，记为“真类缺失”，含义是分数低于导出阈值。
- 相对尺寸需要同图至少 2 架其他飞机，单机或双机图不参与该项。
- 机身旋转会改变轴对齐框的 max(w,h)，同图飞机朝向常一致，相对尺寸部分抵消这一影响。

## 服务器命令（旧服务器，只用 CPU，训练进行中也可运行）

```bash
cd /home/zxy/sar/repos/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
```

工作区有改动或分叉时先查看，不要 reset。不需要准备数据。

找到已有的 test / val 诊断目录（应各有 `DONE`、`baseline/predictions.json.gz`、`rem/predictions.json.gz`）：

```bash
ls -d /data2/zxy/sar/experiments/diagnostics/rem102_comparison_{test,val}_*
export DIAG_TEST=/data2/zxy/sar/experiments/diagnostics/替换为rem102_comparison_test目录
export DIAG_VAL=/data2/zxy/sar/experiments/diagnostics/替换为rem102_comparison_val目录
ls "$DIAG_TEST/DONE" "$DIAG_VAL/DONE"
```

运行（几分钟内结束；`nice` 降低对训练数据加载的影响，不占 GPU）：

```bash
CUDA_VISIBLE_DEVICES= nice -n 19 python tools/analyze_confusion_cues.py "$DIAG_TEST" "$DIAG_VAL" \
  2>&1 | tee "$DIAG_TEST/confusion_cues.console.log"
cat "$DIAG_TEST/confusion_cues/report.md"
cat "$DIAG_VAL/confusion_cues/report.md"
ls -lh "$DIAG_TEST/confusion_cues/crops" "$DIAG_VAL/confusion_cues/crops"
```

确认训练未受影响（`OUT` 为正在运行的 REM + CLCR-Lite 102 轮目录）：

```bash
export OUT=/home/zxy/sar/experiments/替换为正在训练的运行目录
nvidia-smi -i 0,1
tail -n 3 "$OUT/log.txt"
```

打包下载（报告与裁剪图，约几 MB）：

```bash
STAGE=$(mktemp -d)
ln -s "$DIAG_TEST/confusion_cues" "$STAGE/test"
ln -s "$DIAG_VAL/confusion_cues" "$STAGE/val"
tar -czhf ~/confusion_cues_$(date +%Y%m%d_%H%M%S).tar.gz -C "$STAGE" test val
ls -lh ~/confusion_cues_*.tar.gz
```

下载后解压到 `C:\project\实验记录\confusion_cues\`（`test/`、`val/` 各含 `report.md`、`summary.json`、`crops/*.png`），
把两份 `report.md` 贴回，并告知裁剪图位置，以便查看图片。重跑会覆盖 `confusion_cues/` 下的同名输出，
不影响原诊断文件；只要数字不要图时加 `--no-crops`。

## 结果（2026-11-02，102 轮 baseline 与 REM）

来源：用户下载的 `C:\project\实验记录\confusion_cues\{test,val}\`（report.md、summary.json、crops）。
test 焦点对：Boeing737→Boeing787、A220→other、A220→ARJ21、other→A220。

1. **737 是"自信地判错"，不是排序偏差。** test 上 baseline 737 判错 166 个，真类排第 2 的只有 21%，
   与第 1 的分数差中位数 0.705；ARJ21 同样（8.6%，0.675）。A220、other 判错时真类多为第 2（58%、81%），
   属于可校准的一类。分数侧校准（CLCR-Lite 一类）最多改善 A220/other，改不了 737。
2. **尺寸线索完全无效，包括同图相对尺寸。** 各焦点对在源图尺寸、640 尺寸、同图相对尺寸（GT/预测）、
   长宽比上的可分性都在 0.51–0.64。787 实际机长约为 737 的 1.6 倍，但两者源图尺寸中位数相同（91 对 87 像素），
   说明 787 所在图像的真实分辨率更粗；而各类的同图相对尺寸都在 1 左右，推断多数图像是同型号成片停放，
   所以同图相对尺寸在原理上就用不上。**放弃尺寸/相对尺度上下文方向。**
3. **型号判断跟着成像外观走（裁剪图，人工判读）。** 737→787 的错例几乎都是暗、背景干净、散射弱、
   呈细十字轮廓的孤立飞机；判对的 737 则亮、杂波多、停放密集。A220→ARJ21 的错例是高对比的离散亮点簇，
   这正是 ARJ21 判对样本的典型样子；判对的 A220 多为暗弱目标。模型把辐射/成像状态（亮度、对比度、
   亮点密度）当作型号线索，而这种对应关系在 test 上不成立。
4. **737 的错误按图像成簇。** 离散度 1.76，77.6% 的错误来自"该图 ≥80% 的 737 都判错"的图像；
   A220、ARJ21 不成簇（0.92、0.87），是逐个实例的混淆。和第 3 条一致：737 的错误由图像级成像条件决定。
5. **孤立目标在 test 上明显更差，val 上没有这个现象。** test 中同图只有 1–2 架飞机时，定位后型号正确率为 0.543
   （≥11 架时为 0.842），最近邻距离 ≥3 倍自身尺寸时为 0.636；val 上对应为 0.938、0.885。
6. **test 中可能有大量重叠切片。** 裁剪图里多个图像 id 的目标几乎一模一样（A220→ARJ21：#126/#487/#971/#1697/
   #1937/#1990/#2900/#2918/#3023；737→787：#316/#1903/#2021/#4101），像是同一大场景的重叠切片里的同一架飞机。
   若属实，test 的错误高度相关，有效样本量远小于 GT 数，也能解释同一方法在不同协议间 1–3 AP 的波动。
   这一点只是目视判断，尚未定量验证。

**对路线的影响**

- 首选方向：**成像条件一致性。** 训练时对同一图像生成 SAR 风格的辐射与分辨率退化视图（动态范围/gamma、
  对比度、乘性斑点噪声、低通后降采样再放大），让型号判别不依赖辐射状态。先只加增强做消融，验证"成像状态
  shortcut"假设；成立后再加视图间类别一致性约束，作为论文模块（对应文献调研中的"分辨率一致性约束"，
  并扩展到辐射维度）。现有 `RandomPhotometricDistort` 是通用 RGB 抖动，只在 epoch 4–49 生效，
  不模拟 SAR 的动态范围、斑点和分辨率退化。
- 次选方向：**目标内归一化结构几何**（文献 02/03）。737 的暗弱错例仍能看到清晰的十字轮廓，机翼后掠、
  翼展/机长等比例信息是存在的；但需要关键点监督或无监督的形状描述，实现成本更高。
- 原型/对比学习（文献 04）针对的是特征可分性；如果特征仍在编码成像状态，单独使用可能学到同一种 shortcut，
  应排在一致性方向之后，或与之组合。
- 降低优先级：737 的分数校准、相对尺度上下文、拥挤/邻域模块。
- 正在运行的 REM + CLCR-Lite：按第 1、2 条，预期它对 737 没有帮助，可能对 A220/other 有帮助；结果低于 48.3 的可能性大。
- 方法学：这些假设来自 test 诊断。增强参数应按物理含义事先固定，不在 test 上调参；权重仍按 val 选，
  论文中需要说明设计动机来自对测试集错误类型的分析。

## 本地验证

Windows Conda `pytorch`（CPU）：`tests/test_confusion_cues.py` 在合成诊断目录上检查真类名次与分数差、
AUC（含并列）、相对尺寸、离散度和裁剪图输出；`tests/test_type_confusion.py` 仍通过。
未在服务器真实诊断目录上运行，尚无结果。
