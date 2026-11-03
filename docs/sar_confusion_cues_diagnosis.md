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

## 本地验证

Windows Conda `pytorch`（CPU）：`tests/test_confusion_cues.py` 在合成诊断目录上检查真类名次与分数差、
AUC（含并列）、相对尺寸、离散度和裁剪图输出；`tests/test_type_confusion.py` 仍通过。
未在服务器真实诊断目录上运行，尚无结果。
