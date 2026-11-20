# E021：完整模型的 ImageNet 初始化版本（与 E019 配对）

日期：2026-11-19。

## 要回答的问题

E019 显示，在 36 轮协议下 COCO 检测初始化只比 ImageNet 骨干初始化高约 1 AP（val 1.18，test 1.02）。
剩下的问题是：完整模型（E017 = OCEF / E013-v2 + SCRM / E016）相对基线的提升，是否依赖 COCO 检测权重。

| 初始化 | 基线 | 完整模型 | 提升 |
|---|---|---|---|
| COCO 检测权重（`-t`） | E000：val 65.29 / test 62.39 | E017：val 68.81 / test 65.84 | val +3.52 / test +3.45 |
| ImageNet 骨干 | E019：val 64.11 / test 61.37 | **E021** | ? |

## 配置与实现

- 配置：`configs/deim_dfine/deim_hgnetv2_m_sardet100k_imagenet_init_full_model_36e_b32.yml`，包含 E017 配置，只改
  `HGNetv2.pretrained: True` 和 `HGNetv2.local_model_dir: /root/zxy/pretrain/hgnetv2/`（与 E019 相同）。
  用 `INITIAL_WEIGHTS=none` 启动，不传 `-t`；encoder、decoder 和方向融合门控从头初始化。
- 复用 E016/E017 的 v3 拼图计划 `/root/zxy/experiments/region_banks/sardet100k_train_region_dense_sar_v3_seed0_b32`。
- 定向测试 `tests/test_imagenet_init_full_model_e021.py`：E021 相对 E017 只有上述两项不同且取值与 E019 相同；
  E021 相对 E019 改动的配置项，正好等于 E017 相对 E000 改动的配置项，取值也相同。本地与 E017、E019 的配置测试共 8 项通过。

## 放在哪台服务器

放在 8 卡服务器 GPU 4、5：E000、E017、E019 都在这台服务器上，表中四个数字同服务器；v3 计划也只在这台服务器上生成过。
放到旧服务器则需要先在旧服务器上重新生成 v2 和 v3 拼图计划，并补跑一次 ImageNet 基线（E019-old，约 16 小时），
否则 E021 − E019 是跨服务器的差值。

## 协议

与 E017 相同：SARDet-100K 官方 train / val，seed 0，全局 batch 32（GPU 4、5 各 16），36 轮（flat 20，末 4 轮无增强），
AdamW 主 LR 4e-4 / 骨干 4e-5，EMA 0.9999，AMP，RegionDenseSAR v3 代替 Mosaic / MixUp，训练多尺度 480–800、评估 640，每轮完整 val。
唯一区别是初始化。按最高 val AP 选权重，test 训练结束后评估一次，不参与选择。预计约 28 小时（E017 为 27:46:22）。

## 事先定好的判读

记 q = (E021 − E019) / (E017 − E000)，以 val 为主（分母 3.52），test 同样计算（分母 3.45）：

| 结果 | 判读 |
|---|---|
| q ≥ 0.7（val 提升 ≥ 2.46） | 方法增益基本不依赖 COCO 检测初始化 |
| 0.3 < q < 0.7 | 一部分增益依赖 COCO 初始化，两种初始化都报告 |
| q ≤ 0.3（val 提升 ≤ 1.06） | 增益主要依赖 COCO 初始化，论文须以 COCO 初始化为前提表述 |

按类看 OCEF 带来的 bridge / harbor 和 SCRM 带来的 tank / 小目标是否保留。单次运行，按上表直接读。
不为 ImageNet 初始化另调轮数；E019 末轮仍在上升，E021 同样报告逐轮曲线。

## 服务器命令（8 卡服务器）

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
git status --short --branch
git -c pull.rebase=false pull --ff-only origin main
git rev-parse --short HEAD
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_imagenet_init_full_model_e021 -v
GPU_IDS=4,5 EXPERIMENT_ROOT=/root/zxy/experiments INITIAL_WEIGHTS=none RUN_PREFIX=imagenet_full_e021 \
  bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sardet100k_imagenet_init_full_model_36e_b32.yml
```

服务器连不上 GitHub 时，在本地生成 bundle 并传过去：

```bash
git bundle create deim_main.bundle <服务器当前提交>..main
scp -P 20106 deim_main.bundle root@10.11.154.192:/root/zxy/deim_main.bundle
```

```bash
cd /root/zxy/deim && git pull --ff-only /root/zxy/deim_main.bundle main
```

监控与恢复（先设置 `OUT` 为 `STARTED` 行中的目录）：

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
GPU_IDS=4,5 bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

训练结束后按 val 选权重并评估 test，命令与 [E017](sardet100k_full_model_e017.md) 相同；对比可用
`tools/sardet100k/compare_runs.py`（`--run E019=... --run E021=...`）。
