# E003：stride-4 精细层（P2）

日期：2026-11-06。只针对 SARDet-100K。代码 `engine/deim/fine_level.py`，配置 `configs/deim_dfine/deim_hgnetv2_m_sardet100k_fine_level_36e_b32.yml`，测试 `tests/test_fine_level.py`。结果记入[实验总表](sardet100k_results_ledger.md)。

## 为什么选它

1. **按类平均下，权重最大的短板是小 tank 和小 bridge。** tank、bridge 占全类平均 2/6，全局 AP 只有 57.66、45.97。OGSOD 的 tank 和 bridge 只占 GT 的 19.6%，却占漏检的 50.8%（[来源 × 类别评测](sardet100k_source_class_eval.md)）。
2. **这和结构分辨率相符。** 输入长边 16–32 的目标约占 GT 的 13%，AP 50.9、AP75 47.6；长边 64–128 是 73.8、83.8。OGSOD 的 tank 输入长边 16–32 时 AP 32.9，也就是 stride-8 上只有 2–4 格。
3. **冻结检验答不了这个问题。** 冻结小头改变不了特征，还容易学到 val 标注习惯（见 [BAFR](sardet100k_bafr_design.md)），这里必须端到端训练。
4. **它和以前的负结果不同。** 旧的 HWD 和 S2 注入只把细节加进 stride-8，输出仍是 8/16/32，且是在 OGSOD 或 SAR-AIRcraft 上测的。这里新增一层真正的 stride-4 特征，同时参与 query 选择和解码器采样。
5. **它和 GREA 不同。** GREA 只在最后两层解码器把 S2、S3 证据加到 query 上，不产生 stride-4 的 query。

这是标准结构，不是新发明，论文不能把它写成创新点。它要回答的是：更高的结构分辨率能否拿到小目标的空间，并为后面的方法提供基础。

## 模块

```text
原 backbone → 原 encoder → P3, P4, P5（结构和 COCO 权重不变）
S2 ─ 1×1(128) ┐
P3 ─ 1×1(128) ─ 上采样 ×2 ┴ 拼接 → 1×1(256) → 两个残差块（深度卷积 3×3 + 1×1）→ P2
解码器输入：P3, P4, P5, P2（P2 排在最后，原层序号不变）
```

- **query 选择：** P2 的 25,600 个 token 与原 8,400 个 token 一起按分数取前 300。anchor 边长 0.025，是 stride-8 的一半。
- **解码器：** 每层新增 3 个 P2 采样点，用独立的 `fine_offsets`、`fine_weights`，原层形状不变。

## 设置

| 项目 | 值 |
| --- | --- |
| 协议 | 与 E000 相同：36 轮，b32，seed 0，COCO 初始化，同一增强和优化器 |
| 与 E000 的差异 | `HGNetv2.return_idx` 增加 S2；`use_fine_level`、`fine_s2_channels: 96`、`fine_dim: 128`、`fine_points: 3` |
| COCO 权重 | 所有原有参数形状不变，只有 `anchors`、`valid_mask` 重新生成（测试覆盖） |
| 新损失 | 无 |

超参数都按结构设定，没有调过。

## 判定（先于结果写定）

只看 val，test 只在方法定稿后评估一次。

| 项目 | 内容 |
| --- | --- |
| 主指标 | val AP 比 65.29 涨 ≥0.5；不到 1.0 时要补基线和方法的第二种子 |
| 机制指标 1 | tank、bridge AP 合计涨 ≥2，且主要来自 OGSOD |
| 机制指标 2 | 输入长边 <32 的 AP 上升（基线 <16 为 36.6，16–32 为 50.9） |
| 副作用 | APm、APl 下降不超过 0.5。旧 HWD 有 APm −6.5 的先例 |
| 证伪 | AP 涨幅不到 0.5，或小目标不涨 |
| 消融（有信号后） | 只用于 query 选择；只用于解码器采样 |

主观估计（没有实验依据）：涨幅 ≤+0.3 约 40%，+0.3–1.0 约 30%，+1.0–2.0 约 22%，>2 约 8%。

## 局限和风险

- **图像本身的细节有限：** OGSOD 是 256 图放大到 640，P2 不产生新信息，只提高空间分辨率。目标对比度也低，所以有可能不涨。
- **标注精度上限：** OGSOD tank 中位长边约 8 像素，1 像素偏移就会让 IoU 低于 0.75，AP75 的空间可能有限。
- **显存和速度：** 未测。E000 峰值 18.2 GB，P2 还会增加；多尺度到 800 时压力最大。启动后第一步先看显存。
- **边界：** 按现有 `eps=0.01`，P2 最外两圈 anchor 无效，约 8 个输入像素；这一带的目标仍可由其他层选出。

## 服务器命令

8 卡服务器（`-p 20106`），只用 GPU 0、1、4、5。代码路径 `/root/zxy/deim`。

更新代码：

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
```

配置里的数据路径和 E000 相同。8 卡服务器的数据在 `/root/zxy/SARDet_100K`，所以先建链接，配置就不用改：

```bash
mkdir -p /root/zxy/TCRA-SAR/datasets && ln -s /root/zxy/SARDet_100K /root/zxy/TCRA-SAR/datasets/SARDet_100K
```

启动（两张卡都空闲才会启动）：

```bash
EXPERIMENT_ROOT=/root/zxy/experiments INITIAL_WEIGHTS=/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth RUN_PREFIX=deim_m_sardet100k bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sardet100k_fine_level_36e_b32.yml
```

重连：

```bash
export OUT=$(ls -dt /root/zxy/experiments/deim_m_sardet100k_*fine_level* | head -n 1); echo "$OUT"
tail -f "$OUT/console.log"
```

GPU：

```bash
watch -n 2 nvidia-smi
```

指标：

```bash
python -c 'import json,os; rows=[json.loads(s) for s in open(os.environ["OUT"]+"/log.txt") if s.strip()]; r=max(rows,key=lambda x:x["test_coco_eval_bbox"][0]); print(r["epoch"],[round(100*x,2) for x in r["test_coco_eval_bbox"][:6]])'
```

TensorBoard：

```bash
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

恢复（确认进程已停止）：

```bash
cd /root/zxy/deim && bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

停止（只杀 `launcher.pid` 会留下占显存的 worker）：

```bash
kill $(cat "$OUT/launcher.pid"); sleep 5; kill $(ps -eo pid,args | grep train.py | grep "$OUT" | grep -v grep | awk '{print $1}')
```
