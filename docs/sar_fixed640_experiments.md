# SAR-AIRcraft：固定 640 输入实验

本组由用户明确要求固定每个网络输入为 640×640、关闭多尺度。
首个启动配置为 configs/deim_dfine/sar_aircraft_detail/baseline_640.yml，
即原始 baseline 的固定640对照。HGNetv2 仍只返回原有 S3/S4/S5，不加入
S2 注入或其他新模块。B1/B2/B2-L 均不随此任务启动。

## 输入与协议

每个训练样本经过已有 Resize(size=[640,640]) 后输入 collate。
将 collate_fn.base_size_repeat 显式设为 null，使实际 collate.scales=None，
关闭 batch 阶段原有 480–800 的随机尺寸重采样。不能用 0 代替 null，
因为 0 只会取消 640 的重复次数，其他尺度仍存在。

eval_spatial_size 保持 [640,640]。保留 stop_epoch=90，它还与原训练阶段
切换相关；不借助修改 stop_epoch 来关闭多尺度。Mosaic 内部的几何增强
继续沿用原设置，最终网络输入固定为 640×640。

其余继承冻结的 DEIM-M 协议：SAR-AIRcraft-1.0 七类；102 epochs；seed0；
GPU0、1，global batch28（每卡14）；COCO 初始化
/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth；AdamW，主 LR3.5e-4、
backbone LR3.5e-5，WD1e-4，Norm/BN 零衰减，betas(.9,.999)；flatcosine，
warmup2000、flat_epoch49、no_aug_epoch12、lr_gamma.5；AMP fp16、FP32 loss、
原有异常重试，SyncBN 和 EMA 均保持。RandomZoomOut、RandomIoUCrop 关闭。
Mosaic/MixUp/颜色扰动/水平翻转概率及阶段与原协议一致。

baseline 模型和检测损失保持原配置。S2 的固定640变体仅准备供后续选择，
其模型细节另见 [S2 设计](sar_s2_detail_experiments.md)。
训练和验证复用 /root/zxy/datasets/SAR-AIRcraft-1.0 的现有标注，不重新处理数据。

## 对照与判断

当前 baseline 固定640对照旧多尺度 baseline，只改变输入尺寸策略，
用于判断固定640的影响。后续若运行 s2_plain_640.yml，应以这次固定640
baseline 为模块对照，不能把尺度策略与模块变化合并归因。

记录同一最佳 AP checkpoint 的 AP/AP50/AP75/AR100、每类 AP、640 输入尺度组
及相同召回下错误统计。选型使用固定 val，测试集留作最终确认。

## Linux 更新与启动

```bash
cd /root/zxy/deim
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
bash tools/run_sar_module_experiment.sh configs/deim_dfine/sar_aircraft_detail/baseline_640.yml
```

如 worktree 有修改或分歧，先检查并保留，不做强制覆盖。
启动脚本创建全新目录，记录解析配置、Git revision、seed、权重、增强和 GPU 状态。
仅当 GPU0 和1同时空闲才通过 CUDA_VISIBLE_DEVICES=0,1、torchrun 双进程启动；
nohup 后台运行，stdout/stderr=console.log、stdin=/dev/null，保存 launcher.pid 后 disown。

## 日志、指标、恢复和评估

重连后激活 deim_zxy，并显式设置 OUT 为启动命令返回的实际目录：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/替换为实际STARTED目录
tail -n 80 -f "$OUT/console.log"
nvidia-smi -i 0,1
ps -fp "$(cat "$OUT/launcher.pid")"
tail -n 1 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

最佳指标解析代码见 S2 设计文档；本次 baseline 没有 train_s2_* 分支诊断。
AP 数值乘100报告百分数，不能混用不同 epoch 的指标峰值。
eval/latest.pth 是最近评估结果，不一定是最佳。

仅对已停止、需要继续的运行恢复，且恢复脚本仍会检查两卡：

```bash
cd /root/zxy/deim
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

last.pth 只在无强增强阶段前更新，不保证是最后一轮。按评估 AP 和日志选择
best_stg1.pth 或 best_stg2.pth；以 stage2 最佳为例：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/best_stg2.pth" --test-only
```

新训练用 -t COCO 初始化；恢复/评估用 -r，不同时提供二者。
