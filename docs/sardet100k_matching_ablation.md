# 匹配与框监督消融：E004–E006

日期：2026-11-08。只针对 SARDet-100K，DEIM-D-FINE-M，36 轮 E000 协议。结果记入[实验总表](sardet100k_results_ledger.md)。

## 依据

诊断全部基于 E000 的 val，不训练：

- **问题所在：** OGSOD 小而密的 tank 上，最终排在第一的 query 常常不是框最好的那个。可得最佳 IoU 的中位数是 .746，实际选中的只有 .663；GT-aware 去重 Oracle 的上限为 50.2 AP，当前是 30.8。
- **匹配器的选择：** 按训练代码在 val 上重放，Decoder 最后一层有 81% 的对中，框更好的 query 的 L1+GIoU 代价更低，却因为分类代价更高没被匹配。
- **降低分类代价的离线效果：** 分类代价权重 2→1 时，480 对中匹配到更好框的比例从 19.0% 升到 26.7%，原有 IoU ≥ .5 的匹配一个都没丢；MSAR 的匹配完全不变。
- **梯度冲突：** 分类与框梯度在 query 特征上基本正交（余弦中位数 ±.015），没有发现强的负冲突。因此 E005 的依据比 E004 弱。

## 实验

| ID | 改动 | 配置 |
| --- | --- | --- |
| E004（E1） | 匹配器分类代价权重 2 → 1；L1 5、GIoU 2 不变 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_match_cls1_36e_b32.yml` |
| E005（E2） | `DEIMCriterion.use_uni_set: False`：框损失和 FGL/DDF 改用各层自己的匹配，不再用跨层并集 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_layer_box_36e_b32.yml` |
| E006（E3） | 两项同时改 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_match_cls1_layer_box_36e_b32.yml` |

其余设置与 E000 完全相同，已用 `load_config` 逐字段比较确认。Matcher、MAL、LQE 和 Decoder 结构都不改。

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

GPU 0、1 被其他进程占用，所以三个实验依次排队：`tools/run_sar_module_queue.sh` 等前一个训练进程结束、两张卡都空闲后，再调用 `run_sar_module_experiment.sh` 启动下一个。

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

启动排队（按 E004、E005、E006 顺序）：

```bash
cd /root/zxy/deim
Q=/root/zxy/experiments/queue_sardet100k_matching_ablation_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).log
GPU_IDS=4,5 EXPERIMENT_ROOT=/root/zxy/experiments INITIAL_WEIGHTS=/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth RUN_PREFIX=deim_m_sardet100k \
  nohup bash tools/run_sar_module_queue.sh \
  configs/deim_dfine/deim_hgnetv2_m_sardet100k_match_cls1_36e_b32.yml \
  configs/deim_dfine/deim_hgnetv2_m_sardet100k_layer_box_36e_b32.yml \
  configs/deim_dfine/deim_hgnetv2_m_sardet100k_match_cls1_layer_box_36e_b32.yml \
  > "$Q" 2>&1 < /dev/null &
echo $! > "${Q%.log}.pid"; disown; echo "$Q"
```

查看排队和当前运行：

```bash
Q=$(ls -t /root/zxy/experiments/queue_sardet100k_matching_ablation_*.log | head -n 1); tail -n 20 "$Q"
export OUT=$(ls -dt /root/zxy/experiments/deim_m_sardet100k_*{match_cls1,layer_box}* 2>/dev/null | head -n 1); echo "$OUT"
tail -f "$OUT/console.log"
watch -n 2 nvidia-smi
```

指标（AP 最高的一轮）：

```bash
python -c 'import json,os; rows=[json.loads(s) for s in open(os.environ["OUT"]+"/log.txt") if s.strip()]; r=max(rows,key=lambda x:x["test_coco_eval_bbox"][0]); print(r["epoch"],[round(100*x,2) for x in r["test_coco_eval_bbox"][:6]])'
```

TensorBoard：

```bash
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

停止排队和当前运行（只在需要时）：先停排队，避免它接着启动下一个；再停训练。

```bash
kill $(cat "${Q%.log}.pid")
kill $(cat "$OUT/launcher.pid"); sleep 5; kill $(ps -eo pid,args | grep train.py | grep "$OUT" | grep -v grep | awk '{print $1}')
```

恢复中断的运行（确认进程已停止）：

```bash
cd /root/zxy/deim && GPU_IDS=4,5 bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

## 结果

待训练完成后填写。
