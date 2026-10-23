# SAR 数值修复与 batch 28 实验交接

## 已确认的比较条件（2026-10-23）

| 条件 | 原 SAR | 修改后 SAR |
| --- | --- | --- |
| 单卡 batch | 24 | 28 |
| 数值策略 | 原 FP16 | BF16 + FP32 损失/SAR 部分运算 |
| 每轮迭代 | 611 | 523 |
| 第 66 轮 AP | 43.6936% | 40.6977% |

AP 相差约 2.996 个百分点，但 batch 与数值策略同时变化，不能把全部差异归因于某一精度操作或模块设计。新运行每轮少 88 次迭代，约 14.4%；相同 epoch 不等于相同迭代数量。

保持统一的单卡 batch 28。当前 BF16@28 运行可按原计划继续并保留结果；以下 FP16@28 方案是单独的备选，不要求立即追加训练。比较时应统一 batch、数据、初始化、优化器、日程、增强、EMA 和数值策略，并记录实际启动日志。运行中修改 YAML 不会改变已创建的 DataLoader。

## 代码与验证范围

实现提交：1c20989（main，已推送）。

- engine/deim/scale_aware_refinement.py：精度操作开关关闭时恢复原 SAR 的 softmax、RoIAlign、stack 后求和、MLP 和宽高计算。网络层、参数键和初始化顺序保持。
- engine/deim/dfine_decoder.py：向 SAR 传递开关。
- engine/solver/det_engine.py：正常批次使用原 FP16 损失输入。仅非有限输出或 loss 触发一次同批次 FP32 重算，先恢复 BatchNorm buffer 和 PyTorch CPU/CUDA 随机状态；异常前向保存的激活暂存 CPU，不拆分 batch。
- engine/solver/det_solver.py：传递选项并打印运行精度。
- configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_sar_root_fp16_nanfix.yml：独立的新配置。BF16 历史配置及默认运算路径保留。

新配置使用 float16、GradScaler enabled=True、amp_loss_fp32=False、amp_fp32_retry=True、sar_fp32_ops=False、global batch=28。

异常重算打印 AMP_FP32_RETRY、epoch、step、image_ids 和原因；JSONL 增加 train_fp32_retry_rate。FP32 重算仍非有限时立即报错。CPU 暂存增加异常批次的 CPU 内存和传输开销，实际峰值与耗时仍需运行确认。

这不等于已经定位首次失效算子，也没有实训证明 AP 已恢复。仅完成语法解析、配置继承和源码静态核对；未运行模型、回归测试、冒烟测试或训练，未进行文件哈希操作，未在服务器启动训练。

## 保留的协议

静态解析确认新配置与原 SAR 根配置的优化器、DataLoader、增强、EMA 和训练日程相同。

- 单卡 GPU 0；global batch 28；seed 0；120 epoch。
- 基准输入 640，保留原多尺度与增强策略。
- AdamW：基础 LR 0.0004，backbone 0.0002；保留原 scheduler、warmup 和 EMA。
- 初始化：/root/zxy/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth。
- 数据根：/root/zxy/TCRA-SAR/datasets/OGSOD-1.0/sar。
- 标注：RTDETR_COCO/train.json、RTDETR_COCO/val.json。img_folder 仍为 sar 根目录；无需重新准备或改写标注。
- 原验证集、COCO bbox 评估、启用时使用 EMA。

此配置与 BF16 配置比较的是两套数值策略，不是单独一项 BF16 操作的消融。

## 服务器 Git 更新

先查看工作区；有未提交修改或不能快进时，保留现场检查差异，不强制覆盖。

    cd /root/zxy/deim
    git status --short
    git fetch origin
    git switch main
    git pull --ff-only origin main
    conda activate deim_zxy

## 从 COCO 初始化的新运行（选择该方案时执行）

    cd /root/zxy/deim
    conda activate deim_zxy
    CFG=configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_sar_root_fp16_nanfix.yml
    INIT=/root/zxy/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth
    OUT="/root/zxy/experiments/deim_dfine_s_ogsod_120e_1gpu_sar_fp16_nanfix/seed0_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUT"
    git rev-parse HEAD > "$OUT/git_revision.txt"

记录完整配置（只记录，不执行模型）：

    python - "$CFG" "$OUT" "$INIT" <<'PY'
    import sys
    from pathlib import Path
    import yaml
    from engine.core.yaml_utils import load_config
    cfg = load_config(sys.argv[1], {})
    cfg.update(seed=0, use_amp=True, output_dir=sys.argv[2], tuning=sys.argv[3])
    Path(sys.argv[2], "resolved_config.yml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    PY

正式启动：

    CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nproc_per_node=1 train.py \
      -c "$CFG" -t "$INIT" \
      --seed 0 --use-amp --output-dir "$OUT" \
      -u train_dataloader.total_batch_size=28 \
      > "$OUT/console.log" 2>&1 < /dev/null &
    PID=$!
    echo "$PID" > "$OUT/launcher.pid"
    disown
    echo "OUT=$OUT"

应显示 float16、GradScaler enabled=True、FP32 loss=False、FP32 retry=True，以及 building train_dataloader with batch_size=28，新运行从 Epoch: [0] 开始。

## 日志、进程、GPU、指标与 TensorBoard

    tail -f "$OUT/console.log"
    ps -fp "$(cat "$OUT/launcher.pid")"
    nvidia-smi
    tail -n 3 "$OUT/log.txt"
    grep 'AMP_FP32_RETRY' "$OUT/console.log"
    tensorboard --logdir "$OUT/summary" --port 6006

重连后先激活环境，并设置实际完整输出目录：

    cd /root/zxy/deim
    conda activate deim_zxy
    OUT='替换为本次 FP16_nanfix 运行的完整输出目录'

JSONL bbox 指标 0、1、2 是 AP、AP50、AP75，乘 100 后报告百分数：

    python - "$OUT/log.txt" <<'PY'
    import json, sys
    with open(sys.argv[1]) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for row in rows[-5:]:
        ap, ap50, ap75 = row["test_coco_eval_bbox"][:3]
        print(f"epoch={row['epoch']} AP={ap*100:.2f} AP50={ap50*100:.2f} AP75={ap75*100:.2f}",
              "retry_rate=", row.get("train_fp32_retry_rate"))
    PY

## 恢复同一个 FP16_nanfix 运行

先设置 OUT 为实际目录；恢复不同时传 -t。

    cd /root/zxy/deim
    conda activate deim_zxy
    OUT='替换为本次 FP16_nanfix 运行的完整输出目录'
    CFG=configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_sar_root_fp16_nanfix.yml

    CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nproc_per_node=1 train.py \
      -c "$CFG" -r "$OUT/last.pth" \
      --seed 0 --use-amp --output-dir "$OUT" \
      -u train_dataloader.total_batch_size=28 \
      > "$OUT/console_resume_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
    PID=$!
    echo "$PID" > "$OUT/launcher.pid"
    disown

## 最佳权重评估

逐轮指标在 log.txt，TensorBoard 在 summary/，COCO 中间评估数据在 eval/latest.pth。

last.pth 只在无增强阶段前更新；本协议 stop_epoch=112，因此它不是最终 epoch 的保证。best_stg1.pth 与存在时的 best_stg2.pth 按实测 AP 比较，不能按文件名直接判定优劣。

    cd /root/zxy/deim
    conda activate deim_zxy
    OUT='替换为本次 FP16_nanfix 运行的完整输出目录'
    CFG=configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_sar_root_fp16_nanfix.yml

    for NAME in best_stg1 best_stg2; do
      [ -f "$OUT/$NAME.pth" ] || continue
      STAMP=$(date +%Y%m%d_%H%M%S)
      EVAL_OUT="$OUT/evaluate_$NAME"
      CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py \
        -c "$CFG" -r "$OUT/$NAME.pth" --test-only --seed 0 \
        --output-dir "$EVAL_OUT/$STAMP" \
        2>&1 | tee "$EVAL_OUT.$STAMP.log"
    done

