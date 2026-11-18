# E019：ImageNet 初始化基线（量化 COCO 初始化的贡献）

日期：2026-11-17。8 卡服务器 GPU 4、5（与 E000 同一台服务器）。

## 目的

E000 及其后所有实验都用 DEIM 官方 COCO 检测权重 `deim_dfine_hgnetv2_m_coco_90e.pth` 整模型初始化（`-t`），
SARDet-100K 文献多为 ImageNet 骨干初始化，所以绝对数字不能直接与文献比较。E019 把初始化换成只用 ImageNet 骨干，
其余与 E000 完全相同，给出：

- COCO 检测初始化本身的贡献：E000 − E019；
- 与文献同类初始化下的参照行（论文文献参考表）；
- 之后与"完整模型的 ImageNet 初始化版本"配对，检验完整模型的提升是否依赖 COCO 权重。

## 配置与实现

- 配置：`configs/deim_dfine/deim_hgnetv2_m_sardet100k_imagenet_init_36e_b32.yml`，包含 E000 配置，只改
  `HGNetv2.pretrained: True` 和 `HGNetv2.local_model_dir: /root/zxy/pretrain/hgnetv2/`。
- 骨干权重：`/root/zxy/pretrain/hgnetv2/PPHGNetV2_B2_stage1.pth`（24,362,501 字节），即
  `https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth`，
  D-FINE / DEIM 官方 COCO 训练的起点。encoder 和 decoder 随机初始化。
- 启动脚本 `tools/run_sar_module_experiment.sh` 新增 `INITIAL_WEIGHTS=none`：新运行不传 `-t`，
  启动前检查配置开启了 `HGNetv2.pretrained` 且 stage1 文件存在，并在 `run_metadata.json` 记录初始化方式。
  其他运行的命令不变（`-t` / `-r` 的构造已核对）。
- 定向测试 `tests/test_imagenet_init_e019.py`：配置相对 E000 只有上述两项不同，骨干冻结设置为 `freeze_at=-1`、`freeze_norm=False`；
  `pretrained=True` 时从 `local_model_dir` 读入 stage1 权重。

## 协议

与 E000 相同：SARDet-100K 官方 train / val，seed 0，全局 batch 32（GPU 4、5 各 16），36 轮（flat 20，末 4 轮无增强，warmup 2000 步），
AdamW 主 LR 4e-4 / 骨干 4e-5、WD 1e-4，EMA 0.9999，AMP，Mosaic / MixUp / Dense O2O 与 E000 相同，训练多尺度 480–800、评估 640，每轮完整 val。
按最高 val AP 选权重，test 在训练结束后评估一次，不参与选择。

不为 ImageNet 初始化另调轮数或学习率：比较对象是同一协议下的差值。从 ImageNet 开始 36 轮可能尚未收敛，
这本身是 COCO 初始化贡献的一部分；逐轮 val 曲线一并报告。

## 读法

- E000 val 65.29 / test 62.39。E000 − E019 记为 COCO 检测初始化的贡献。
- E019 与文献数字并列时，仍要注明训练轮数（36）、输入（640）和检测器（DEIM-D-FINE-M）与文献不同。
- 完整模型的 ImageNet 初始化版本（下一项，同服务器）与 E019 的差值，和 E017 − E000 比较，判断方法增益是否依赖 COCO 权重。

## 服务器命令

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
git status --short --branch
git -c pull.rebase=false pull --ff-only origin main
git rev-parse --short HEAD
GPU_IDS=4,5 EXPERIMENT_ROOT=/root/zxy/experiments INITIAL_WEIGHTS=none RUN_PREFIX=imagenet_e019 \
  bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sardet100k_imagenet_init_36e_b32.yml
```

日志里应出现 `Loaded stage1 B2 HGNetV2 from local file.`。监控与恢复（先设置 `OUT` 为实际运行目录）：

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

恢复时用 `-r last.pth`，构建模型时仍会先读入 stage1 骨干，再被 checkpoint 覆盖，不影响结果。

## 启动记录

2026-11-17 23:40（服务器时间）在 8 卡服务器 GPU 4、5 启动，代码 `1ada691`（服务器连不上 GitHub，经 git bundle 快进，提交号与 GitHub 一致）。
启动前服务器上 `tests.test_imagenet_init_e019` 通过；stage1 文件为 382 个张量的 state_dict。launcher PID `2759406`，运行目录：

```bash
export OUT=/root/zxy/experiments/imagenet_e019_deim_hgnetv2_m_sardet100k_imagenet_init_36e_b32_seed0_20261117_234029_hGExcb
```

`start_command.sh` 中没有 `-t`；`run_metadata.json` 记录 initialization 为 `/root/zxy/pretrain/hgnetv2/PPHGNetV2_B2_stage1.pth`，
模式为 "ImageNet HGNetv2 stage1 backbone only (no -t); encoder/decoder from scratch; fresh optimizer"。
日志有 `Loaded stage1 B2 HGNetV2 from local file.`（只在 rank 0 打印；两个进程读同一本地文件）。
第 0 轮第 100 步：每步约 0.55 s，峰值显存每卡 14.8 GB，与 E000 相同，无报错。

## 结果（2026-11-18）

36 轮完成，训练耗时 17:39:05（E000 为 17:37:45）。最高 val AP 在第 36 轮（index 35），选 `best_stg2.pth`（`last_epoch=35`），EMA 权重。
test 为官方 11,613 张图，640 输入，GPU 4、5，全局评估 batch 64，top-300，maxDets=100；test 未参与选择。
评估目录 `$OUT/test_eval_selected_29fujN`（`eval.pth`、`metrics.json`、`comparison.json`、`selection.json`、`summarize_results.py`）。

| 划分 | 初始化 | AP | AP50 | AP75 | APs | APm | APl | AR100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| val | E000：COCO 检测权重（`-t`） | 65.29 | 93.15 | 70.26 | 60.18 | 76.63 | 72.09 | 73.36 |
| val | **E019：ImageNet 骨干** | 64.11 | 92.41 | 69.14 | 58.78 | 75.59 | 69.12 | 72.58 |
| val | 差值 | −1.18 | −0.74 | −1.12 | −1.40 | −1.04 | −2.97 | −0.78 |
| test | E000：COCO 检测权重（`-t`） | 62.39 | 90.92 | 66.35 | 59.01 | 74.65 | 71.72 | 70.88 |
| test | **E019：ImageNet 骨干** | 61.37 | 89.64 | 65.58 | 58.50 | 73.75 | 67.93 | 70.22 |
| test | 差值 | −1.02 | −1.28 | −0.77 | −0.51 | −0.90 | −3.79 | −0.66 |

| 类别 | val E019 | val 差值 | test E019 | test 差值 |
|---|---:|---:|---:|---:|
| ship | 73.05 | −0.69 | 71.93 | −0.97 |
| aircraft | 61.86 | −0.92 | 48.62 | −1.18 |
| car | 71.02 | −0.19 | 71.72 | −0.18 |
| tank | 56.54 | −1.13 | 53.61 | −0.30 |
| bridge | 45.06 | −0.90 | 42.99 | −1.88 |
| harbor | 77.11 | −3.29 | 79.38 | −1.57 |

逐轮 val AP（E019 / E000）：第 1 轮 29.37 / 36.39，第 5 轮 42.30 / 47.40，第 10 轮 44.63 / 49.01，第 20 轮 51.54 / 54.71，
第 30 轮 61.56 / 63.05，第 33–36 轮 62.95、63.46、63.83、64.11 / 64.48、64.85、65.11、65.29。差距从第 1 轮的 7.0 缩小到第 36 轮的 1.2；
两者末轮仍在小幅上升（E019 +0.28，E000 +0.18）。

判读（单次运行）：

- **在本协议下，COCO 检测初始化只贡献约 1 AP（val 1.18，test 1.02）。** E000 的高分主要来自检测器（DEIM-D-FINE-M）、36 轮训练和 640 输入，
  不是 COCO 权重。只用 ImageNet 骨干时 test 仍为 61.37，比 SARDet-100K 文献中 ImageNet 初始化的最高报告值（DenoDet V2 56.71，12 轮，
  MMDetection，评测划分未说明）高 4.7；这一差距不能归因于 COCO 初始化，但仍包含检测器、训练轮数和输入尺寸的差异。
- **COCO 初始化帮得最多的是大目标。** APl 差 2.97（val）/ 3.79（test），明显大于 APs 的 1.40 / 0.51；按类 harbor（val −3.29）最大。
- **差距随训练缩小，末轮仍略有上升。** 36 轮下 E019 可能尚未完全收敛，更长训练可能进一步缩小差距；这属于 COCO 初始化贡献的一部分，按协议不另调轮数。
- 论文用途：作为初始化对照表中的"ImageNet 初始化基线"，以及文献参考表中与文献同类初始化的参照行。
  完整模型的 ImageNet 初始化版本尚未运行（本次按用户要求不启动新实验）。
