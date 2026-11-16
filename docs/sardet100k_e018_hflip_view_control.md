# E018：水平翻转视图对照（E013-v2 的机制对照）

日期：2026-11-16。8 卡服务器 GPU 4、5。

## 要回答的问题

E013-v2（原图 + 垂直翻转视图，投影空间门控融合）在 E000 上 val +1.97、test +2.35。这个增益可能来自两种原因：

1. 垂直翻转视图带来了原图没有的上下方向证据（论文想主张的机制，依据 D17、D18）；
2. 只是"同一张图再过一次 backbone，再做门控融合"的通用双视图效果，与方向无关。

E018 只把第二个视图换成水平翻转，结构、参数量、计算量、初始化、训练协议全部与 E013-v2 相同。
训练数据本来就有随机水平翻转（`RandomHorizontalFlip`），模型对左右方向已基本不敏感，所以水平翻转视图几乎不提供新的方向信息，
但保留了"第二次编码 + 门控融合"的全部其余因素。

## 配置与实现

- 配置：`configs/deim_dfine/deim_hgnetv2_m_sardet100k_orientation_fusion_v2_hflip_view_36e_b32.yml`，
  包含 E013-v2 配置，只把 `DEIM.orientation_fusion.views` 从 `['vflip']` 改为 `['hflip']`。
- 代码：`engine/deim/orientation_projection_fusion.py` 原来只接受 `vflip`，现在接受 `vflip` 或 `hflip` 中的一个。
  默认仍是 `vflip`，E013-v2 和 E017 的行为、日志键名（`orientation_vflip_p*`）不变；E018 的日志键名为 `orientation_hflip_p*`。
- 网格补偿：沿用 3 像素 / 1 格。HGNetv2 stem 在右侧和下侧的填充相同，
  `tests/test_orientation_fusion.py` 的校准测试已对 `vflip` 和 `hflip` 两个方向都检查过（对齐误差 < 5%，不补偿时 > 30%）。
  平移产生的无效列（第 0 列）同样被掩码，门控只能选原图。
- 定向测试 `tests/test_orientation_fusion_v2_hflip.py`：
  配置相对 E013-v2 只有 `views` 不同，训练仍含随机水平翻转；水平视图的输入、对齐和无效列掩码正确；
  日志键名；E018 与 E013-v2 的参数名、形状和同种子初始值完全一致（新增参数同为 101,808）。
  本地与现有方向融合测试共 22 项通过。

## 协议

与 E013-v2 相同：SARDet-100K 官方 train / val，seed 0，全局 batch 32（GPU 4、5 各 16），COCO-M 初始化
`-t /root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth`，36 轮（flat 20，末 4 轮无增强），AdamW 4e-4 / backbone 4e-5，
EMA 0.9999，AMP，Mosaic / MixUp / Dense O2O 与 E000 相同，训练多尺度 480–800、评估 640，每轮完整 val。
按最高 val AP 选权重；test 只在训练结束后评估一次，不参与选择。预计训练约 27–28 小时。

## 事先定好的判读

以 val AP 为主，记 r = (E018 − E000) / (E013-v2 − E000) = (E018 − 65.29) / 1.97：

| 结果 | 判读 | 对论文的影响 |
|---|---|---|
| r ≤ 0.3（val ≤ 65.9） | 增益主要来自上下方向证据 | 支持主张；E018 作为论文中的关键对照 |
| 0.3 < r < 0.7 | 一部分是通用双视图效果，一部分来自方向 | 两部分都如实报告，方向的贡献按差值计 |
| r ≥ 0.7（val ≥ 66.7） | 增益主要是通用双视图效果 | 论文论点改为双视图融合，不再主张上下不对称 |

同时看按类 AP：E013-v2 的增益集中在 bridge（val +7.30）和 harbor（+4.15）。若 E018 在这两类上也有相近增益，同样按"通用双视图"解释。
单次运行；判读按上表直接读，不另加种子。

## 服务器命令

见本文末尾"启动记录"一节的实际运行目录。更新代码与启动：

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
git status --short --branch
git fetch origin && git -c pull.rebase=false pull --ff-only origin main
git rev-parse --short HEAD
GPU_IDS=4,5 EXPERIMENT_ROOT=/root/zxy/experiments \
INITIAL_WEIGHTS=/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth \
RUN_PREFIX=hflipview_e018 bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sardet100k_orientation_fusion_v2_hflip_view_36e_b32.yml
```

服务器连不上 GitHub 时，在本地生成 bundle 传过去再快进：

```bash
git bundle create deim_main.bundle <服务器当前提交>..main
scp -P 20106 deim_main.bundle root@10.11.154.192:/root/zxy/deim_main.bundle
```

```bash
cd /root/zxy/deim && git pull --ff-only /root/zxy/deim_main.bundle main
```

监控（先设置 `OUT` 为实际运行目录）：

```bash
tail -f "$OUT/console.log"
```

```bash
nvidia-smi -i 4,5; ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
```

```bash
python -c 'import json,os; [print(r["epoch"]+1, [round(100*x,2) for x in r["test_coco_eval_bbox"][:6]]) for r in map(json.loads, filter(str.strip, open(os.environ["OUT"]+"/log.txt")))]'
```

```bash
tensorboard --logdir "$OUT/summary" --host 127.0.0.1 --port 6006
```

中断后恢复（不与 `-t` 同时使用）：

```bash
GPU_IDS=4,5 bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

训练结束后按 val 选权重并评估 test：选择规则和命令与 [E017](sardet100k_full_model_e017.md) 相同（`best_stg1` / `best_stg2` 中按 log.txt 的最高 val AP 对应的那个），
test 图像 `/root/zxy/TCRA-SAR/datasets/SARDet_100K/JPEGImages/test`，标注 `/root/zxy/TCRA-SAR/datasets/SARDet_100K/Annotations/test.json`。
