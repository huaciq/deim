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

## EMA 重放：训练过程中的匹配变化

脚本 `tools/sardet100k/tank_matching_dynamics.py`：对一个运行的每个 checkpoint（每 4 轮一个，外加 `final.pth`），用 EMA 权重在固定的 OGSOD、MSAR val tank 图上重放该运行自己的匹配器。统计直接调用 `DEIMCriterion._record_matching_metrics`，与训练日志的定义完全相同。在 CPU 上运行，每个 checkpoint 约 2 分钟。

**口径说明：** 每个 Decoder 层恰好匹配 GT 数个 query，而它们都在并集里，所以“框为正、分类为背景”的数量在每一层都等于“并集大小 − GT 数”，逐层相同。下表因此用“每个 GT 多出的并集 query 数”来表示这类 query，用“连续 ≥ 2 层 / 全部 4 层为背景”来表示它们持续多久。

### E000 基线（OGSOD tank 1,216 个 GT；MSAR tank 522 个）

| 轮 | OGSOD 匹配 IoU（dec4 均值 / ≥ .75） | OGSOD 4 层同一 query | dec1→dec2 / dec2→dec3 / dec3→dec4 同一 query | OGSOD 每 GT 多出的框正分类背景 query | 连续 ≥ 2 层背景 / 全部 4 层背景（占并集） | MSAR 匹配 IoU | MSAR 4 层同一 query | MSAR 每 GT 多出 |
|---:|---|---:|---|---:|---|---:|---:|---:|
| 3 | .498 / .139 | .822 | .880 / .918 / .935 | .260 | .221 / .085 | .857 | .994 | .010 |
| 7 | .512 / .151 | .844 | .887 / .930 / .964 | .242 | .211 / .092 | .864 | .996 | .008 |
| 11 | .535 / .187 | .852 | .892 / .918 / .975 | .229 | .204 / .088 | .868 | .996 | .004 |
| 15 | .555 / .214 | .851 | .895 / .924 / .963 | .257 | .224 / .097 | .875 | .998 | .002 |
| 19 | .570 / .226 | .819 | .878 / .921 / .961 | .278 | .240 / .089 | .882 | 1.000 | .002 |
| 23 | .601 / .257 | .798 | .841 / .924 / .963 | .347 | .277 / .112 | .902 | .998 | .002 |
| 27 | .629 / .286 | .813 | .856 / .927 / .969 | .336 | .275 / .116 | .926 | .998 | .006 |
| 31 | .641 / .318 | .834 | .864 / .945 / .972 | .316 | .254 / .116 | .944 | .998 | .008 |
| 35 | .645 / .317 | .836 | .877 / .932 / .969 | .294 | .246 / .106 | .957 | .998 | .004 |

读法：

- **OGSOD 层间不一致的位置与大小：** 匹配在各层间换 query 的情况主要发生在 Decoder 1→2，同一 query 的比例只有 .84–.89。“框为正、分类为背景”的 query 每个 GT 多出 .23–.35 个，在第 23 轮附近最多；第 20 轮正是 Mosaic/MixUp 结束、学习率开始下降的时候。
- **MSAR 几乎没有这个现象。**
- **第 19–27 轮，后几层匹配到的框反而不如 Decoder 1：** 匹配 IoU 均值 dec1 > dec4，例如第 23 轮为 .606 对 .601，与“后层按分类重新选择”一致；到第 35 轮又略有回升。
- **E004 若机制成立，应看到：** OGSOD 的 4 层同一 query 比例上升，每 GT 多出的并集 query 减少，匹配 IoU 与 IoU ≥ .75 的比例上升；MSAR 基本不变。

结果在 E000 运行目录的 `independent_verify_20261106/matching_dynamics_E000/report.json`。

E004 的重放（checkpoint 生成后运行；输出目录必须直接放在 E000 的 `independent_verify_20261106/` 下，脚本从那里读取来源分组）：

```bash
cd /root/zxy/deim && export PATH=/opt/miniconda3/bin:$PATH && source /opt/miniconda3/etc/profile.d/conda.sh && conda activate deim_zxy
E0=/root/zxy/experiments/deim_m_sardet100k_deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop_seed0_20261105_125147_N2zAJh
C=$E0/independent_verify_20261106/matching_dynamics_E000/code
OUT=$(ls -dt /root/zxy/experiments/deim_m_sardet100k_*match_cls05* | head -n 1)
O=$E0/independent_verify_20261106/matching_dynamics_E004
mkdir -p $O && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 PYTHONPATH=$PWD nohup nice -n 10 python $C/tank_matching_dynamics.py $OUT $O --device cpu > $O/replay.log 2>&1 < /dev/null &
echo $! > $O/replay.pid; disown
```

脚本会跳过 `report.json` 里已经算过的 checkpoint，所以可以在训练过程中反复运行。脚本本身及其依赖还没有提交到 Git，运行时用的是上面目录里的副本。

## 结果

待训练完成后填写。
