# SAR-AIRcraft-1.0：DEIM-D-FINE-M 固定协议

配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_102e_b28.yml`。
本次只固化协议和核对权重加载，不启动训练，不停止正在运行的 SARDet baseline。
原 S/120e 配置及其结果保留。

## 参数取值

| 参数 | 固定值 |
| --- | --- |
| 模型 | 官方 DEIM-D-FINE-M；HGNetv2-B2，4 层 decoder，7 类 |
| COCO 初始化 | /root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth；train.py -t |
| 数据 | 原 train=3047、val=442、test=879；不合并为 trainval，不把 test 当验证集 |
| 基础 / 验证输入 | 640×640；保留官方训练多尺度 |
| GPU / batch | GPU 0、1；global train batch=28，每卡 14 |
| 验证 batch | 官方 global 64，每卡 32 |
| 优化器 | AdamW，betas=(0.9,0.999)，WD=1e-4；Norm/BN WD=0 |
| 主 LR | 3.5e-4 = 官方 4e-4 × 28/32 |
| Backbone 非 BN 组 LR | 3.5e-5 = 官方 4e-5 × 28/32 |
| 轮数 | 102，日志 epoch 0..101 |
| LR scheduler | 官方 flatcosine，二次 warmup=2000 steps，flat_epoch=49，lr_gamma=0.5 |
| 最小 LR | 主组 1.75e-4；backbone 非 BN 组 1.75e-5 |
| 强增强边界 | [4,49,90]；epoch=90 起进入最后 12 轮收尾 |
| 关闭增强 | RandomZoomOut、RandomIoUCrop 全程关闭；其余沿用官方 M |
| 精度 / seed | AMP FP16 + GradScaler，seed=0 |
| EMA / norm | EMA decay=0.9999、warmups=1000；BN 不冻结、SyncBN、clip norm=0.1 |

数值策略明确保留项目已有 FP32 loss 与非有限批次一次 FP32 重算，属于运行层面的
已有修复，不是新增模块或损失；以后同协议 baseline 和方法实验保持这一设置。

## 官方来源与口径补充

[官方 M 配置](https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/deim_dfine/deim_hgnetv2_m_coco.yml)
给出主 LR=4e-4、backbone 非 BN LR=4e-5、102 epochs、flat=49、no_aug=12、
policy=[4,49,90]。本配置按用户指定的 28/32 比例缩放两组 LR，保持优化器分组正则不变。
[官方 DEIM 基础配置](https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/base/deim.yml)
给出 warmup=2000、Mosaic/MixUp 等设置。

官方参数组先选 backbone 且名称不含 bn 的参数为低 LR，再选 Norm/BN 为 WD=0。
因此 **backbone BN 也属于 Norm/BN 组，使用主 LR=3.5e-4、WD=0**；
不能将“backbone LR=3.5e-5”理解为所有 backbone 参数都使用低 LR。

按 3047 张、两卡每卡 14、drop_last，训练每轮 108 steps，合计 11016 steps。
2000 步 warmup 约为 **18.52 个 epoch**；保留官方步数，不按小数据集自动缩短。
flat_iter=5292，余弦下降至 step=9720，最后 1296 steps 维持最小 LR。

权重文件名里的 90e 指 COCO 预训练检查点，迁移训练从 epoch 0 新建优化器训练 102 轮，
不是从 COCO epoch 90 恢复后只训练 12 轮。使用 -t 初始化；-r 仅用于后续中断恢复。
七类编号按已准备的 category_map.json 为 0..6，COCO 的 80 类分类头不能直接作为七类头。
由现有加载流程匹配通用参数，形状不兼容的分类/去噪类别参数保留新模型初始化。

### 实际权重与参数组核对

服务器 CPU 已按现有 BaseSolver.load_tuning_state 加载用户给定的 M COCO 权重，
文件大小 79,143,599 bytes。HGNetv2-B2、4 层 decoder、分类头 [7,256]、去噪类别嵌入
[8,256] 均确认。形状不兼容的七类头保留 seed=0 下的新初始化，通用权重按现有代码匹配加载。
没有将 COCO 前七类当作本数据集类别，没有运行 forward 或 optimizer step。

| 官方参数组 | 参数张量数 | 参数个数 | LR | WD |
| --- | ---: | ---: | ---: | ---: |
| backbone 非 BN | 142 | 5,999,930 | 3.5e-5 | 1e-4 |
| Norm/BN（含 backbone BN） | 296 | 59,464 | 3.5e-4 | 0 |
| 其余参数 | 206 | 13,418,175 | 3.5e-4 | 1e-4 |

核对输出保存在
`/root/zxy/experiments/protocol_checks/deim_m_sar_aircraft_20261026_124427_HbzX8x/`，
本地副本在 `C:/Users/DELL/.codex/artifacts/sar_aircraft_deim_m_protocol_20261026/`。
本次完成配置解析、与官方 M 的受控字段比较及 LR 边界验证；未进行训练、吞吐或显存测量。

## 增强口径

用户已明确确认：**关闭 RandomZoomOut、RandomIoUCrop，其余沿用官方 M**。
两项已从 transforms.ops 和 policy.ops 中移除，保留：

- Mosaic 外层 p=0.5，内部 probability=1.0；MixUp p=0.5，均在 4≤epoch<49 开启。
- PhotometricDistort p=0.5，在 4≤epoch<90 使用。
- 水平翻转 p=0.5；训练多尺度 480..800、步长 32，640 在 16 个候选中占 6 个。
- epoch 90..101 停止强增强和多尺度，固定 640；普通水平翻转继续按官方实现保留。

640 在本协议中是基础/验证尺寸，并非全程禁止多尺度。增强停止点是零起始日志
epoch=90，即完成前 90 轮之后进入 12 轮收尾。

此次确认只调整上述两项增强及对应 policy 列表；102 轮、batch 28、两组学习率、
2000 步预热、[4,49,90] 调度和 480–800 多尺度保持不变。配置合并后已核对，剩余操作
及其参数与官方 M 完全一致。权重加载和参数组未受影响，沿用此前 CPU 核对结果。

## Git、环境和未来训练命令（本次不执行训练）

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
```

服务器 GitHub HTTPS 无法连接时，通过 Git bundle 的 fetch 和 --ff-only merge 同步，
保留现有工作区文件。数据已经准备好，不需要重新转换。

未来启动时创建新目录并冻结实际配置：

```bash
CONFIG=configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_102e_b28.yml
OUT=$(mktemp -d /root/zxy/experiments/deim_m_sar_aircraft_102e_b28_seed0_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_XXXXXX)
export CONFIG OUT
git rev-parse HEAD > "$OUT/git_revision.txt"
cp "$CONFIG" "$OUT/source_config.yml"
cp /root/zxy/datasets/SAR-AIRcraft-1.0/COCO/stats.json "$OUT/dataset_stats.json"
python - <<'PY'
import json,os,pathlib,yaml
from engine.core.yaml_utils import load_config
p=pathlib.Path(os.environ['OUT']);c=load_config(os.environ['CONFIG'],{})
c.pop('__include__',None);c.update(output_dir=str(p),summary_dir=str(p/'summary'),seed=0)
(p/'resolved_config.yml').write_text(yaml.safe_dump(c,sort_keys=False))
(p/'run_metadata.json').write_text(json.dumps(dict(git_revision=(p/'git_revision.txt').read_text().strip(),seed=0,devices=[0,1],global_batch=28,initialization='/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth',initialization_mode='COCO tuning (-t), no resume',resolved_config=c),indent=2))
PY
ulimit -Sn 65535
nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -t /root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth \
  --seed 0 --use-amp --output-dir "$OUT" \
  > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
tail -f "$OUT/console.log"
```

今后中断时恢复，只针对已停止的独立运行执行：

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/替换为实际M运行目录
LOG="$OUT/resume_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).console.log"
ulimit -Sn 65535
nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/last.pth" --seed 0 --use-amp --output-dir "$OUT" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
tail -f "$LOG"
```

当前 solver 的 last.pth 只在 epoch<90 更新，正常最后为 epoch 89，不是最终权重。
比较 best_stg1/best_stg2 的实际验证 AP 再选权重；不要凭阶段名选择。

日志/指标/GPU/TensorBoard（重连后激活环境并明确 OUT）：

```bash
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
watch -n 2 nvidia-smi
tail -f "$OUT/console.log"
tail -n 3 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

test_coco_eval_bbox 前三项是 AP/AP50/AP75，原值 0..1，报告乘 100。
训练逐轮评估 val.json；选好权重后最终 test.json 评估命令：

```bash
STAGE=best_stg1
EVAL=$(mktemp -d "$OUT/test_${STAGE}_XXXXXX")
ulimit -Sn 65535
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/$STAGE.pth" --test-only --use-amp \
  --output-dir "$EVAL" --summary-dir "$EVAL/summary" \
  -u val_dataloader.dataset.ann_file=/root/zxy/datasets/SAR-AIRcraft-1.0/COCO/test.json \
  > "$EVAL/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$EVAL/launcher.pid"
disown "$PID"
tail -f "$EVAL/console.log"
```
