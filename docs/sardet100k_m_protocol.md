# SARDet-100K 主基线：DEIM-D-FINE-M，P0（36 轮）与 P4（102 轮）

用户决定只跑这两组，不做种子、输入尺寸和 ZoomOut/IoUCrop 的校准（2026-11-04）。
尚无结果。论文对比行不引用文献数字，用同协议自跑。

## 配置

| 运行 | 配置 | 轮数 | flat | 增强边界 | MixUp | 多尺度停止 | 无强增强 |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| P0 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop.yml` | 36 | 20 | [4,20,32] | [4,20] | 32 | 4 |
| P4 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_102e_b32_nozoom_nocrop.yml` | 102 | 49 | [4,49,90] | [4,49] | 90 | 12 |

两组只有日程不同。P4 是公开 M 配置的日程；P0 按同一规则缩短（32 + 4，flat = 4 + 32 // 2）。

| 项目 | 设置 |
| --- | --- |
| 数据 | 官方 train 94,493 张 / val 10,492 张；六类，类别 ID 0..5，`remap_mscoco_category: False` |
| 初始化 / 种子 | `-t /root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth`；seed 0 |
| 批量 | 训练全局 32（每卡 16），2952 steps/epoch；验证全局 64 |
| 输入 | 基础和验证 640；训练多尺度 480..800，步长 32 |
| 优化器 | AdamW，主 LR 4e-4，骨干 4e-5，WD 1e-4，norm/BN WD 0，clip norm 0.1 |
| 学习率 | flatcosine，warmup 2000 步，lr_gamma 0.5 |
| 增强 | Mosaic、MixUp 各 p=0.5，PhotometricDistort p=0.5，水平翻转 p=0.5；关闭 RandomZoomOut、RandomIoUCrop |
| EMA / AMP | EMA 0.9999，warmups 1000；FP16，FP32 loss，非有限批次一次 FP32 重算 |
| 评估 | 每轮完整 val，COCO bbox |

`total_batch_size` 已是全局值。显存按已有记录粗算峰值约 22 GB，未实测。

## 决策规则（提议，待用户确认）

- P4 与 P0 的 AP 差不超过 1：后续模块筛选和正式实验都用 36 轮。
- 差超过 1：模块筛选仍用 36 轮，正式实验用 102 轮。
- 没有 P1，单种子波动未测，因此模块筛选用固定门槛：36 轮单种子涨 0.5 AP 以上才晋级，
  曲线结束时明显高于基线并仍在上升的补一次长日程。最终方法和基线再跑多种子。

## 启动前

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
```

确认标注目录里有没有 test 标注。官方 README 称只发布 train、val；没有 test 时正式结果只报告 val，并在论文写明。

```bash
ls -la /root/zxy/TCRA-SAR/datasets/SARDet_100K/Annotations/
```

## 启动

脚本只在 GPU 0、1 都空闲时启动，不排队，也不会停止其他任务。先跑 P0：

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
EXPERIMENT_ROOT=/root/zxy/experiments INITIAL_WEIGHTS=/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth RUN_PREFIX=deim_m_sardet100k bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop.yml
```

P0 结束后跑 P4：

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
EXPERIMENT_ROOT=/root/zxy/experiments INITIAL_WEIGHTS=/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth RUN_PREFIX=deim_m_sardet100k bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sardet100k_102e_b32_nozoom_nocrop.yml
```

每次新运行都有独立目录。配置、Git revision、种子、权重和 GPU 状态记录在目录内的
`run_metadata.json`、`resolved_config.yml`、`git_revision.txt`、`gpu_gate.json`。

## 日志和 GPU

重连后设置 `OUT`。P4 把 `36e` 换成 `102e`：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
export OUT=$(ls -dt /root/zxy/experiments/deim_m_sardet100k_*_36e_b32_nozoom_nocrop_seed0_* | head -n 1); echo "$OUT"
tail -f "$OUT/console.log"
```

另开终端看进程和 GPU。GPU-Util 长期明显上不去时，先把 `num_workers`（现为每个 dataloader 4）调高再重启；这不改变协议：

```bash
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
nproc
watch -n 2 nvidia-smi
```

## 指标

```bash
python - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ['OUT']) / 'log.txt'
rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
for name, r in [('best', max(rows, key=lambda r: r['test_coco_eval_bbox'][0])), ('latest', rows[-1])]:
    print(name, 'epoch', r['epoch'], dict(zip(['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'],
          [round(100 * x, 3) for x in r['test_coco_eval_bbox'][:6]])))
PY
```

```bash
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

COCO 指标 0、1、2 是 AP、AP50、AP75，原值 0..1，上面已乘 100。首轮验证结束后才有 `log.txt`。

## 评估检查点

`best_stg1.pth` 和 `best_stg2.pth`（如果存在）都要评估，按实测 AP 选，不看阶段文件名。这里评估的是 val：

```bash
cd /root/zxy/deim
STAGE=best_stg1
EVAL=$(mktemp -d "$OUT/eval_${STAGE}_XXXXXX")
ulimit -Sn 65535
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py -c "$OUT/resolved_config.yml" -r "$OUT/$STAGE.pth" --test-only --use-amp --output-dir "$EVAL" --summary-dir "$EVAL/summary" 2>&1 | tee "$EVAL/console.log"
```

## 恢复

只在该运行已停止时执行。脚本用保存的配置和 `last.pth`，并再次检查两张卡是否空闲。
`last.pth` 只更新到无强增强阶段之前（P0 到 epoch 31，P4 到 epoch 89），不是最终轮权重。

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```
