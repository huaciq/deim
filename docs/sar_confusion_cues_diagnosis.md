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
   长宽比上的可分性都在 0.51–0.64。各类的同图相对尺寸都在 1 左右，推断多数图像是同型号成片停放，
   同图相对尺寸在原理上就用不上。**放弃相对尺度上下文方向。**
   但"787 所在图像分辨率更粗"的推断不成立（见下文 SAR 特性复核）：A330 与 A320/737 的框尺寸比例符合实际机长，
   只有 787 不符，问题在 787 这一类本身。
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

**SAR 特性复核（上面的分析把图像当普通灰度图处理，以下几点改变了解释）**

- **787 的框尺寸与实际机长不符。** 源图像素 max(w,h) 中位数：A330 105（test）/ 131（val），A320/321 68，
  737 88 / 90，ARJ21 80 / 69，787 87 / 94。A330 相对 737/A320 约 1.2–1.5 倍，符合宽体机与窄体机的比例；
  787 机长与 A330 相近，却和 737 一样大，只有 p90（125 / 132）接近 A330。可能原因：
  (a) 787 机身和机翼大量使用复合材料，后向散射可能比铝制机体弱，标注框只覆盖到可见的强散射部分；
  (b) 787 类中混有窄体机的标注错误。两者都未验证。
- **"暗弱 → 787"可能部分是真实的物理线索。** 如果 (a) 成立，弱散射本身就和 787 相关，模型学到它并不全错；
  A220 判对的样本多为暗弱目标，而 A220 也使用复合材料机翼，方向一致。错误出在成像整体偏暗、信噪比低的 737
  也被当成弱散射。因此不能用随机亮度/对比度抖动把亮度信息抹掉，应该做**相对场景背景的辐射归一化**
  （目标与周围杂波的对比度），保留目标相对强度、去掉图像级的明暗差异。
- **图像级明暗可能来自产品量化。** SAR 幅度动态范围很大，转成 8 位时的截断/拉伸方式因景而异，会造成整景偏暗或偏亮。
  737 错误按图像成簇与此一致。
- **方位角依赖。** 发动机进气道、翼身结合处、尾翼的二面角/角反射只在特定雷达视角下形成亮点；
  "离散亮点簇 vs 细十字"的差别也可能来自飞机朝向与雷达视线的夹角，而不只是成像条件。
  因此图像旋转增强在物理上不严格（雷达照射方向随图像一起转），现有 Mosaic 旋转 ±10° 与水平翻转也会改变阴影/叠掩方向。
- **斑点是乘性的。** 退化增强应在幅度或对数域用乘性 Gamma 斑点，不用加性高斯；距离向与方位向分辨率不同，
  分辨率退化应各向异性。
- **COCO RGB 预训练。** SAR 单通道复制为三通道，`RandomPhotometricDistort` 中的饱和度和色调抖动在灰度图上不起作用，
  实际只有亮度和对比度。

待验证（服务器 CPU，只读标注与图像）：同图同时出现 787 与 737、A330 与 A320 时的框尺寸比（物理上约 1.6）；
每图背景电平、饱和像素比例、斑点等效视数在 train/val/test 间的分布；每个目标的目标/杂波对比度按类别、
按判对判错的分布；由框内强散射估计的主轴朝向与判错的关系；重叠切片的定量检查。

### SAR 特性检查工具

`tools/analyze_sar_characteristics.py` 读原始 COCO 标注与原图（train/val/test），可选读入已有诊断目录的
baseline/REM 预测，只用 CPU，输出到新目录（已存在则报错，不覆盖）：

| 节 | 内容 | 用来判断 |
|---|---|---|
| 1 | 同图同时含两类时的 max(w,h) 比：787/737、A330/A320、A330/737、787/A330、A220/ARJ21、737/A320 | 787/737 若接近 1 而 A330/A320 约 1.5 → 787 框偏小或标签有误；787/A330 应接近 1 |
| 2 | 每图背景电平（GT 外像素中位数）、p99、饱和/零值比例、斑点等效视数，按划分与源图尺寸 | train/val/test 的量化拉伸是否不同 |
| 3 | 每个目标的 TCR（框内最亮 10% 均值 / 周围杂波环中位数，dB），按划分与类别 | 787（及 A220）是否系统性地散射更弱 |
| 4 | 诊断划分上判对/判错/漏检的 TCR 与背景；按整图背景、TCR（train 三分位）和强散射主轴方向分组的型号正确率 | 错误是否跟着整图明暗、目标对比度或朝向走 |
| 5 | 24×24 框内裁剪的归一化相关（尺寸差 ≤10%，不同图像）找重复目标：各划分去重数、与 train 的匹配数、baseline 判错实例去重数 | 重叠切片有多少；test/val 是否与 train 重叠 |

`targets.csv`、`images.csv` 保留逐目标、逐图数值，便于后续再分析。主轴方向只在轴比 ≥1.5 时计入（十字形飞机常无确定主轴）。
像素按存储值处理，dB 假设为幅度图。

服务器命令（旧服务器；更新代码与环境激活同上）：

```bash
export DIAG_TEST=/data2/zxy/sar/experiments/diagnostics/替换为rem102_comparison_test目录
export DIAG_VAL=/data2/zxy/sar/experiments/diagnostics/替换为rem102_comparison_val目录
export SARCHK=/data2/zxy/sar/experiments/diagnostics/sar_characteristics_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
nohup env CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 nice -n 19 python tools/analyze_sar_characteristics.py \
  --data-root /home/zxy/sar/datasets/SAR-AIRcraft-1.0 --output "$SARCHK" \
  --diag "$DIAG_TEST" --diag "$DIAG_VAL" \
  > "$SARCHK.console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$SARCHK.pid"
disown "$PID"
tail -f "$SARCHK.console.log"
```

进度：依次打印 `measured train/val/test`，最后打印 `report.md` 路径。重连后查看：

```bash
export SARCHK=/data2/zxy/sar/experiments/diagnostics/替换为sar_characteristics目录
ps -p "$(cat "$SARCHK.pid")" -o pid,etime,args
tail -n 5 "$SARCHK.console.log"
cat "$SARCHK/report.md"
```

中断后换一个新的 `SARCHK` 重跑即可（不续算，不覆盖）。打包下载：

```bash
tar -czf "$SARCHK.tar.gz" -C "$(dirname "$SARCHK")" "$(basename "$SARCHK")"
ls -lh "$SARCHK.tar.gz"
```

解压到 `C:\project\实验记录\sar_characteristics\`。

### SAR 特性检查结果（2026-11-03）：val/test 与 train 大量重复，推翻了前面的解释

来源：`/data2/zxy/sar/experiments/diagnostics/sar_characteristics_20261103_110801`
（用户下载到 `C:\project\实验记录\sar_characteristics\`）。以下"匹配"指 24×24 框内裁剪的归一化相关 ≥0.9、
框尺寸差 ≤10%、来自不同图像；"判对"指 score≥0.3、IoU≥0.5 的定位后型号正确（不是 AP）。

**1. 数据集由重叠切片构成，val/test 的大部分目标在 train 中出现过。**

| 划分 | GT | 去重后的飞机数 | 在 train 中有匹配的 GT |
|---|---:|---:|---:|
| train | 11551 | 1051（每架中位数出现 10 次，最多 45 次） | — |
| val | 1962 | 384 | 1604（82%） |
| test | 2950 | 381 | 1485（50%） |

train 内部的重复组没有一组标签不一致；val、test 各有 1 组混有 787 与 other（val 5 个 GT，test 26 个）。

**2. val–test 差距几乎完全由重复比例决定，与成像条件无关。**

| 划分 | 子集 | GT（飞机数） | baseline 定位率 | baseline 型号正确率 | REM 定位率 | REM 型号正确率 |
|---|---|---:|---:|---:|---:|---:|
| val | train 中有匹配 | 1604（288） | 0.983 | 0.978 | 0.984 | 0.984 |
| val | 无匹配 | 358（96） | 0.592 | 0.467 | 0.587 | 0.467 |
| test | train 中有匹配 | 1485（220） | 0.944 | 0.940 | 0.936 | 0.955 |
| test | 无匹配 | 1465（161） | 0.541 | 0.411 | 0.496 | 0.377 |

见过的飞机几乎全对；没见过的飞机，val 和 test 表现相近（型号正确率 0.47 / 0.41）。
val 有 82% 是见过的，test 只有 50%，这就是 val AP50 88 对 test 67 的主要原因。
前文"test 分布不同/更难"和"成像状态 shortcut"的解释不成立，至少不是主因。

**3. REM 的 test 增益来自见过的目标，在没见过的目标上反而更差。** 无匹配子集上 REM 的定位率 0.541→0.496，
定位且判对的比例 0.223→0.187；有匹配子集上 0.888→0.894。这是 score≥0.3 的点统计、单种子，不是 AP，
需要用排除匹配目标的 AP 复核，但方向已经和"REM 是稳定增益"矛盾。

**4. 有效样本量很小。** test 无匹配子集只有 161 架不同的飞机（737 40、other 59、ARJ21 23、A220 21、787 19，
无 A320/A330）；val 无匹配只有 96 架。test 的 166 个 737 判错只来自 23 架飞机。1 AP 量级的差异在这个规模上无法区分。

**5. 前面的分组结论需要作废或重做。** 737 错误按图成簇、按背景/TCR/朝向分组的差异（如 737 在 0° 主轴只有 0.115）
都由少数重复飞机主导，不能当作成像规律。同图尺寸比也不可靠：train 中 A330/A320 只有 1.02（物理上约 1.6），
样本来自少数几景，787 框尺寸与材料/标注的假设既不能支持也不能排除。

**与官方论文核对（王智睿等，雷达学报 2023，SAR-AIRcraft-1.0）**

本地数据与论文完全一致：图像 4368 = 3047 + 442 + 879（约 7∶1∶2），目标 16463 = 11551 + 1962 + 2950；
论文表 3 的 test 各类数量（A330 31、A320/321 52、A220 460、ARJ21 362、737 550、787 454、other 1041）与本地 test 逐类相同，
"训练集"各类数量正好等于本地 train + val（如 A330 278 = 251 + 27，ARJ21 825 = 447 + 378）。
论文说明数据来自高分三号聚束模式 1 m 单极化，取自上海虹桥、北京首都、台湾桃园 3 个机场的多时相影像，
按图像以 7∶1∶2 划分，没有说明按场景隔离，与检查到的重叠切片一致。

论文的检测基准（表 5，IoU=0.5，101 点插值，与 COCO AP50 定义接近）：Faster R-CNN mAP 76.1，SA-Net 77.7，
Boeing737 在所有方法中都最低（54.5–65.1），说明 737 难是数据集本身的特点。我们的 test AP50 为 65–68，低于论文的 Faster R-CNN；
可能原因包括论文的识别实验使用 train + val（13513）训练，检测实验的训练集合与评估细节未写明。未核实。

**对论文和路线的影响**

用户决定（2026-11-03）：**官方数据集与官方划分不做任何改动**，训练、选权重、报告都沿用官方协议，与已发表工作一致。
重复问题是官方划分本身的属性，所有方法面对的条件相同，比较仍然公平。它只用于内部判断：

- 未见/见过子集的拆分只作为诊断，帮助判断某个模块的 test 增益是来自更好的泛化还是更好地记住见过的目标，
  不替代官方指标，也不改变训练数据或选权重方式。是否在论文中作为补充分析，后面再定。
- test 有效样本小（约 381 架不同飞机），1 AP 量级的差异受种子影响大；主结论需要多种子确认。
- val 与 train 重叠更多，val 的排序对 test 的预测力有限（已观察到 Spearman 0.63），选模块时以 test 多种子结果为准。
- 下面"对路线的影响"一节基于重复检查之前的诊断，需结合上述情况重新评估。

**对路线的影响（重复检查之前的判断，暂缓）**

- 首选方向：**成像条件一致性。** 训练时对同一图像生成 SAR 风格的辐射与分辨率退化视图（动态范围/gamma、
  乘性斑点噪声、各向异性低通后降采样再放大），并配合相对场景背景的辐射归一化，让型号判别不依赖图像级明暗、
  同时保留目标相对强度。具体形式等上面的 SAR 特性检查有结果后再定。先只加增强做消融，验证"成像状态
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
`tests/test_sar_characteristics.py` 在合成斑点场景上检查同图尺寸比、TCR 排序、判错实例的结果分组、
重复目标去重与 train 匹配、主轴方向以及输出目录不覆盖。
未在服务器真实诊断目录上运行，尚无结果。
