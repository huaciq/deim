# 匹配与框监督消融：E004–E006

日期：2026-11-08。只针对 SARDet-100K，DEIM-D-FINE-M，36 轮 E000 协议。结果记入[实验总表](sardet100k_results_ledger.md)。

## 依据

诊断全部基于 E000 的 val，不训练：

- **问题所在：** OGSOD 小而密的 tank 上，最终排在第一的 query 常常不是框最好的那个。可得最佳 IoU 的中位数是 .746，实际选中的只有 .663；GT-aware 去重 Oracle 的上限为 50.2 AP，当前是 30.8。
- **匹配器的选择：** 按训练代码在 val 上重放，Decoder 最后一层有 81% 的对中，框更好的 query 的 L1+GIoU 代价更低，却因为分类代价更高没被匹配。
- **降低分类代价的离线效果：** 分类代价权重 2→1 时，480 对中匹配到更好框的比例从 19.0% 升到 26.7%，原有 IoU ≥ .5 的匹配一个都没丢；MSAR 的匹配完全不变。
- **梯度冲突：** 分类与框梯度在 query 特征上基本正交（余弦中位数 ±.015），没有发现强的负冲突。因此 E005 的依据比 E004 弱。

## 实验

| ID | 改动 | 配置 | 状态 |
| --- | --- | --- | --- |
| E004（E1） | 匹配器分类代价权重 2 → **0.5**；L1 5、GIoU 2 不变 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_match_cls05_36e_b32.yml` | 运行 |
| E005（E2） | `DEIMCriterion.use_uni_set: False`：框损失和 FGL/DDF 改用各层自己的匹配，不再用跨层并集 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_layer_box_36e_b32.yml` | 暂停，未运行 |
| E006（E3） | E1 与 E2 同时改 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_match_cls1_layer_box_36e_b32.yml`（分类代价仍为 1，重新启用前要改成与 E1 相同的 0.5） | 暂停，未运行 |

- **只改了目标字段：** 其余设置与 E000 完全相同，已用 `load_config` 逐字段比较确认。MAL 及其他损失权重、LQE 和 Decoder 结构都不改。
- **所有匹配阶段同步：** 训练时只有一个 matcher 对象，Encoder Top-K、Decoder 1 前置头和 Decoder 1–4 的匹配都调用它，所以分类代价在 6 个阶段同时生效。去噪 query 走固定分配，不经过 matcher。
- **系数 1 的那次运行（2026-11-08）：** 配置 `..._match_cls1_36e_b32.yml`，用户决定改用 0.5 后，在第 1 轮内停止。运行目录 `deim_m_sardet100k_deim_hgnetv2_m_sardet100k_match_cls1_36e_b32_seed0_20261108_183430_XLvsRU` 留作记录，内有 `STOPPED.txt`，没有结果。

### 训练中的匹配统计（仅记录，不影响训练）

E004 打开了 `log_matching_stats: True`、`matching_stats_class: 3`（tank）。

- **采样：** `matching_stats_interval: 10`，每 10 步记录一次（每轮约 295 个 batch）。每步都记录时训练从约 0.55 s/步慢到 0.74 s/步，所以那次运行在第 0 轮内停止，目录 `..._match_cls05_36e_b32_seed0_20261108_194743_9I6Jlh` 留作记录，内有 `STOPPED.txt`。
- **统计内容：** criterion 在被采样的训练 batch 上记录真实分配的统计，写入 `log.txt`（键名前缀 `train_match_`，为每轮的平均值）和 TensorBoard（`Module/match_*`）。
- **不影响训练：** 统计在无梯度下计算，不消耗随机数，不进入损失；测试 `tests/test_matching_stats.py` 确认开关打开与关闭时，各项损失逐项相同。
- **对照：** E000 没有这项记录，所以训练中的变化只能和 EMA 重放的结果对照；要直接比较，需要带记录重跑基线。

| 键名 | 含义 |
| --- | --- |
| `gt`、`union` | 本 batch 的 GT 数；框损失并集里的（query, GT）对数 |
| `iou_<阶段>`、`iou50_<阶段>`、`iou75_<阶段>` | 各阶段（enc、pre、dec1–dec4）实际匹配 query 的 IoU 之和，以及 IoU ≥ .5 / ≥ .75 的个数 |
| `same_dec_all`、`same_dec_<k>_<k+1>` | 4 个 Decoder 层都匹配到同一 query 的 GT 数；相邻两层匹配到同一 query 的 GT 数 |
| `boxpos_clsbg_dec<k>` | 在并集中（受框监督）、但在第 k 层未匹配（MAL 当背景）的 query 数 |
| `clsbg_run2`、`clsbg_all` | 并集 query 中，连续 ≥ 2 层 / 全部 4 层被当作背景的个数 |

- **后缀 `_tank`：** 只统计 tank GT。
- **怎么算比例：** 用两个键的每轮均值相除，例如 `train_match_iou_dec4_tank / train_match_gt_tank`。

E005 的注意事项：关闭并集后，框损失的归一化从并集匹配对数变为 GT 数，参与回归的 query 也会变少。所以 E005 同时改变了监督集合和框损失的有效尺度，这是这个改动本身带来的，结论里要写明。

## 判定（先于结果写定）

| 项目 | 内容 |
| --- | --- |
| 协议 | 36 轮，seed 0，全局 batch 32，两卡；只看 val，test 不参与 |
| 主指标 | val AP 相对 E000（65.29）涨 0.5 以上；单种子，不到 1.0 时要补第二个种子 |
| 机制指标 1 | OGSOD tank AP75 高于 E000（17.24，第 9 节口径） |
| 机制指标 2 | 用 `tank_matcher_audit.py` 在新权重上重放：OGSOD 每个 GT 排名第一的 query 中，“另一个 query 好 ≥ .05 IoU”的比例低于 E000 |
| 机制指标 3 | 用 `tank_layer_tracking.py`：OGSOD 最后一层分类 logit 与 IoU 的 Spearman 高于 .244 |
| 不伤害 | MSAR tank AP 不低于 E000 1 个点以上（95.19） |
| 证伪 | 主指标涨幅不到 0.5；或 AP 上涨但机制指标都不变，说明增益与这里的机制无关 |
| 归因 | E004 与 E005 分开判读；E006 只用于看两者能否叠加，不单独作为方法依据 |

局限：E004 只是改一个匹配权重，本身不是方法。只有在机制成立时，才考虑把它发展成针对小目标、考虑定位质量的匹配方法。

## 服务器命令（8 卡服务器，只用空闲的 GPU 4、5）

GPU 0、1 被其他进程占用。目前只跑 E004，直接用 `run_sar_module_experiment.sh` 启动。`tools/run_sar_module_queue.sh` 留给以后需要排队时使用。

更新代码：

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
export PATH=/opt/miniconda3/bin:$PATH
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
```

启动 E004（两张卡都空闲才会启动；启动脚本内部已用 nohup、保存 `launcher.pid` 并 disown）：

```bash
cd /root/zxy/deim
GPU_IDS=4,5 EXPERIMENT_ROOT=/root/zxy/experiments INITIAL_WEIGHTS=/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth RUN_PREFIX=deim_m_sardet100k bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sardet100k_match_cls05_36e_b32.yml
```

查看运行：

```bash
export OUT=$(ls -dt /root/zxy/experiments/deim_m_sardet100k_*match_cls05* | head -n 1); echo "$OUT"
tail -f "$OUT/console.log"
watch -n 2 nvidia-smi
```

训练中的匹配统计（逐轮，tank）：

```bash
python -c 'import json,os; rows=[json.loads(s) for s in open(os.environ["OUT"]+"/log.txt") if s.strip()]
for r in rows:
    g=r["train_match_gt_tank"]; u=r["train_match_union_tank"]
    print(r["epoch"], "IoU dec4 %.3f" % (r["train_match_iou_dec4_tank"]/g), "same4 %.3f" % (r["train_match_same_dec_all_tank"]/g),
          "boxpos_clsbg_dec4 %.3f" % (r["train_match_boxpos_clsbg_dec4_tank"]/u), "bg_run2 %.3f" % (r["train_match_clsbg_run2_tank"]/u))'
```

指标（AP 最高的一轮）：

```bash
python -c 'import json,os; rows=[json.loads(s) for s in open(os.environ["OUT"]+"/log.txt") if s.strip()]; r=max(rows,key=lambda x:x["test_coco_eval_bbox"][0]); print(r["epoch"],[round(100*x,2) for x in r["test_coco_eval_bbox"][:6]])'
```

TensorBoard：

```bash
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

停止运行（只在需要时）：

```bash
kill $(cat "$OUT/launcher.pid"); sleep 5; kill $(ps -eo pid,args | grep train.py | grep "$OUT" | grep -v grep | awk '{print $1}')
```

恢复中断的运行（确认进程已停止）：

```bash
cd /root/zxy/deim && GPU_IDS=4,5 bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

## 结果

待训练完成后填写。
