# REM 102 轮：旧服务器统一导出与配对诊断

## 目的与输入

用户报告 REM / 102 轮 AP65.2、AP50 88.6、AP75 74.1、APm64.0、APl58.5。
原 baseline / 102 轮约 AP64.5、AP75 72.9、APm64.4、APl57.9。
本次先区分整体收益、高 IoU 收益与中目标损失的来源，再决定 REM 的结构改进。
不重新训练，不修改原模型、原保存配置、原标注或 checkpoint。

baseline 已从新服务器转到旧服务器。本文默认转移目录为：

`/data2/zxy/sar/experiments/imported_baseline102_LMw1Yy`

其中应有 best_stg2.pth、log.txt、resolved_config.yml；git_revision.txt 和
run_metadata.json 若存在则归档。若实际目录不同，只替换命令中的 BASELINE。
原 baseline 记录最佳逻辑 epoch90、AP64.510414；脚本仍依据保存日志的最高验证 AP
选择 best_stg1.pth / best_stg2.pth，不以 last.pth 或 checkpoint 内部 epoch 代替选择。

REM 来源：

`/data2/zxy/sar/experiments/deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_rem_102e_b28_old_server_seed0_20261101_145427_rtK11d`

用户已确认两服务器的数据集相同，直接使用旧服务器现有
`/home/zxy/sar/datasets/SAR-AIRcraft-1.0/COCO/val.json` 和对应图像。
无需迁移 val.json/test.json、重新转换标注或再传一份图片。

## 新增工具及行为

- tools/compare_rem_102e.py：归档输入、选择最佳权重、生成独立推理配置、顺序双卡导出、CPU 配对分析和打包。
- tools/run_rem_102e_diagnosis.sh：新建诊断目录、nohup 后台启动、保存 PID、支持中断恢复。
- tests/test_rem_102e_diagnosis.py：针对权重选择、配置适配、配对几何、COCO 计算和恢复的功能验证。

原 resolved_config.yml 按原字节保存在诊断目录，另生成 inference_config.yml。
推理副本仅关闭骨干单独预训练下载、设置旧服务器验证图像/标注路径和导出目录，
以及导出 batch（每卡8、全局16）。训练 batch28、模型开关、损失和训练协议保留归档。
两组都使用 checkpoint 的 EMA（若保存）、FP32、固定640、原 top300 后处理，
无置信度预过滤、无新增 NMS。先 baseline、再 REM，依次占用 GPU0、1，不并行启动两组。

## 更新与一条命令启动

先查看工作树；有本地改动或分叉时先保留并处理，不强制覆盖。GPU0、1 可用于导出后执行：

```bash
cd /home/zxy/sar/repos/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy

export BASELINE=/data2/zxy/sar/experiments/imported_baseline102_LMw1Yy
export REM=/data2/zxy/sar/experiments/deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_rem_102e_b28_old_server_seed0_20261101_145427_rtK11d
export DATA_ROOT=/home/zxy/sar/datasets/SAR-AIRcraft-1.0
export DIAGNOSIS_ROOT=/data2/zxy/sar/experiments/diagnostics
nvidia-smi -i 0,1
bash tools/run_rem_102e_diagnosis.sh "$BASELINE" "$REM"
```

脚本通过 nohup 启动后台 Python 工作进程，stdout/stderr 保存 console.log，stdin
来自 /dev/null；先保存 $! 到 launcher.pid 再 disown。每个导出由工作进程调用
torchrun --standalone --nproc_per_node=2。它不排队、不停止已有任务。
STARTED 表示工作进程已启动；实际导出状态见 EXPORT_START / EXPORT_DONE，
两组完成后自动进行 CPU 分析并打印 PACKAGE 路径，最终写入 DONE。

## 查看进度与结果

将 OUT 替换为启动器打印的诊断目录；重连时重新激活环境并显式设置 OUT：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/diagnostics/替换为实际诊断目录
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
nvidia-smi -i 0,1
tail -f "$OUT/console.log"
```

更细的导出进度分别在 baseline/export.console.log 和 rem/export.console.log：

```bash
tail -n 20 "$OUT/baseline/export.console.log"
tail -n 20 "$OUT/rem/export.console.log"
```

Ctrl+C 退出跟踪后查看 CPU 报告：

```bash
cat "$OUT/report/report.md"
```

该诊断不产生新的训练 TensorBoard 事件；完整逐轮数据在 report/epoch_metrics.csv，
原训练事件仍在各自原训练目录的 summary/，不需要为第一轮诊断额外迁移。

## 中断后的恢复

只在原诊断进程结束后执行：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/diagnostics/替换为原诊断目录
bash tools/run_rem_102e_diagnosis.sh --resume "$OUT"
```

恢复日志是打印的 resume_*.console.log。已成功完成、带 export.done 标记的导出被复用；
未完成的一组重新导出，再生成报告和结果包。此操作是恢复诊断，不使用 train.py -r
恢复训练，也不需要 COCO 初始化权重。

## 下载与诊断定义

完成后只需下载 `$OUT/diagnosis_bundle.tar.gz`，其中包含两组完整预测、原始保存配置和
推理副本、训练日志、权重选择记录、共同验证标注，以及以下报告；不包含权重或图片。

- report.md / summary.json：同口径复算指标、与日志 AP 差值、逐 IoU AP、阶段均值与错误计数。
- class_metrics.csv：七类别 AP/AP75/APm/APl 及 REM 相对 baseline 的变化。
- gt_pairs.csv / paired_groups.csv：同一 GT 在 IoU .50/.75/.85/.90 下，两者都检出、新增、损失、都未检出。
- common_gt_geometry.csv：共同检出 GT 的 IoU/分数变化、中心、宽高与四边误差。
- candidate_quality.csv：全量导出候选的最佳同类/任意类 IoU，以及高 IoU 正确类和错类的最高分数。
- epoch_metrics.csv：两组全部记录轮次的 AP/AP50/AP75/APs/APm/APl，以及 REM alpha（若日志记录）。

AP 使用全分数预测与官方 faster-coco-eval，缺少 GT 的尺度/类别指标记为 n/a。
新增/损失计数固定 score>=0.3、同类按分数一对一匹配，按 image_id/gt_id 配对，不按 Query 编号。
这些阈值计数不是 AP 损失归因。共同检出几何统计以两组 score>=0.3、IoU>=0.5 为条件，
报告有符号偏差和绝对误差；候选最佳 IoU 是逐 GT 覆盖分析，不是一对一召回率。
所有结果均使用同一个旧服务器验证集；第一轮不执行 test 推理或进行模型调参。

如果将结果包下载本机后需要重新计算 CPU 报告，可在解压目录上运行：

```powershell
conda run --no-capture-output -n pytorch python tools/compare_rem_102e.py analyze --output '替换为解压目录'
```

## 本地验证

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_rem_102e_diagnosis.py -v
```

覆盖最佳验证 AP 权重选择、原配置保留及旧路径适配、同 GT 得失/错类/低分候选、
共同检出几何、完整分数 COCO AP 与缺失尺度、报告/不含权重的打包，以及恢复阶段路由。
本地使用合成数据验证统计定义，未执行服务器真实权重的双卡导出，不将合成结果当作 REM 实验结果。

本地 7 项功能验证全部通过。
