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
