# SAR-AIRcraft：SLQC / DSG 实验准备与交接

本次完成模块、独立配置与运行入口的准备，不自动启动新训练，也不改写已有 baseline 或 SCS 运行。
SLQC 和 DSG 是待验证的候选机制；代码接入、功能正确性与训练收益分别记录。
所有候选保持原 DEIM-M 七类检测、COCO M 初始化与固定训练协议。

## 实验矩阵

配置目录：`configs/deim_dfine/sar_aircraft_modules/`。

| 配置 | 在固定 baseline 上增加 | 用途 |
|---|---|---|
| `dsg.yml` | DSG，directional | 三个方向高频分别门控 |
| `dsg_shared.yml` | DSG，shared | 共享高频门控的消融 |
| `dsg_ll_only.yml` | DSG，ll_only | 低频残差分支的消融 |
| `dsg_spatial.yml` | DSG，spatial | 同类空间残差/门控对照 |
| `slqc.yml` | SLQC，layout | 局部散射布局引导 Query 分类 |
| `slqc_roi.yml` | SLQC，roi | 只有局部 ROI 信息，无显式布局编码 |
| `scs_dsg.yml` | 原 SCS + DSG | 描述监督与子带调制的组合 |
| `scs_slqc.yml` | 原 SCS + SLQC | 描述监督与局部布局交互的组合 |
| `slqc_dsg.yml` | SLQC + DSG | 两个推理时分支的组合 |
| `scs_slqc_dsg.yml` | SCS + SLQC + DSG | 完整组合，独立机制通过后再比较 |

每项有独立的 `output_dir`；正式启动器还会创建带时间和随机后缀的新运行目录。
配置采用单父级继承：非 SCS 实验继承冻结 baseline，SCS 组合继承完整 M1 或其子配置，
不会因多重 include 顺序把 SCS 损失权重、teacher collator 或类别对设置覆盖掉。

建议先比较独立模块和相应对照，再试组合；没有预设哪个配置一定涨点。
DSG四种模式均为60,353参数，`spatial`只匹配参数，计算量/时延不相同。
SLQC的layout/roi分别为84,098/79,298参数；roi去掉4,800参数的位置编码器，
因此它是内容对照，不宣称严格等参数。额外SCS仍为37,020参数。

## 模块位置与边界

### SLQC

SLQC 从最终普通解码 Query 及其预测框出发，在 P3/P4 上采样局部 token，
以归一化位置/关系编码补充局部布局。输出分类 Query 的残差，交给原分类头权重产生 logit 增量。
原 LQE 质量项保留；原 FDR 框回归不改为使用分类残差。
浅层定位蒸馏继续使用原始最终层分类分数作为teacher，避免同时改变DDF权重。
ROI 坐标取预测框的 detached 值；推理不需要 GT、不新增部件关键点标签。
SCS 组合仍单独监督原最终 Query 的描述，DN 与普通 Query 的索引不能混用。

`layout` 与 `roi` 比较用于判断：增益是否来自显式布局，而非额外局部信息或参数。
不把局部 token 先命名为机头、翼尖、发动机语义 token。

### DSG

DSG 接在 Hybrid Encoder 内 S3/stride8 的 256 通道投影之后、金字塔融合之前。
使用通道瓶颈、固定 Haar 分解、频带条件门控、逆变换与残差。
保持原三个特征尺度，不新增 P2，不替换 COCO 预训练 stem。
方向细节的符号/布局不能先被解释成纯噪声。
`ll_only`只限制新增残差分支使用LL；原特征恒等路径仍保留高频，不是将整网/输入低通。
`shared`只共享三个高频门控，子带滤波与能量条件仍分方向，不能称为完全去掉方向信息。
初始残差为恒等映射；只将残差标量置零，不把残差最后投影同时置零。

这些设计借鉴已有原则，不能将“小波+门控”或普通 ROI attention 包装成首次提出的物理方法。
相关原始来源包括 [PGD](https://arxiv.org/abs/2411.12301)、
[SFS-Conv](https://github.com/like413/SFS-Conv)、
[WTConv](https://arxiv.org/abs/2407.05848) 与
[WSF-YOLO](https://www.mdpi.com/2072-4292/18/18/3254)。
它们提供结构/频率融合的研究依据，并不证明本次配置已经获得收益。

## 固定训练协议

| 项目 | 所有新增配置沿用的设置 |
|---|---|
| 数据 | SAR-AIRcraft-1.0；train3047 / val442 / test879；7类 |
| 数据根 | `/root/zxy/datasets/SAR-AIRcraft-1.0` |
| 标注 | 原 `COCO/train.json`、`COCO/val.json`，不用重新生成 |
| 模型 | DEIM-D-FINE-M，HGNetv2-B2，4层解码器 |
| 初始化 | `/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth`，新训练用 `-t` |
| GPU / batch | GPU0、1；全局28，每卡14 |
| seed / epochs | 0 / 102，日志 epoch0–101 |
| 输入 | 基础/验证640；前90轮保留480–800多尺度，step32 |
| 优化器 | AdamW，主LR3.5e-4、backbone非BN3.5e-5、WD1e-4、betas(.9,.999) |
| Norm/BN组 | 主LR，WD0，保留官方参数分组 |
| 调度 | flatcosine、warmup2000 steps、flat49、末12轮、最小LR比例.5 |
| 增强 | RandomZoomOut / RandomIoUCrop关闭；其他沿用已固定M协议 |
| 强增强调度 | `[4,49,90]`；Mosaic/MixUp在4≤epoch<49；90起固定640 |
| 数值 | AMP FP16、GradScaler、FP32 loss、非有限批次FP32重算 |
| 其他 | EMA(.9999,warmups1000)、SyncBN、clip norm .1 |
| 验证 | 原后处理top300、不额外NMS、不按score预截断AP；val batch64 |

主比较使用 val 选权重，test 留给最终评估。不要用 test 重新选结构、门控或置信度。
640输入时，DSG四种频带模式中的directional/shared/ll_only卷积MAC解析估计约0.253G/图，
spatial约0.384G/图，不含变换、归一化、能量和逐点运算；不是GPU时延测量。
标准AMP保持参数FP32，两个新模块内部运算FP32后回到输入dtype；不支持显式整模块.half()权重。
GPU显存、吞吐及训练收益尚未测量，不能把预算当实际性能。

## 功能验证记录

已验证Haar符号/重建及奇数尺寸、子带对照隔离、ROI中心/批次/层级顺序、布局敏感性、
框坐标detach、AMP FP32路径、零初始化及学习梯度。整网接入回归检查公共初始化与seed保持，
alpha=0的检测输出、alpha非零时仅最终分类变化、LQE/DN/aux定位teacher保持，
eval/deploy无需GT且保留SLQC，原检测损失与SCS组合均能回传到新增参数。
本地共28项针对性回归通过（DSG6、SLQC8、整网/配置8、原SCS6），使用Conda pytorch / PyTorch2.5.1。
10项配置逐项核对原训练协议；本次不进行正式训练或额外启动前试跑。
修复配置读取的共享默认字典，连续加载不同实验不再遗留前一个实验的模块开关；
原冻结配置文件未改，单次配置解析行为保持一致。
日志新增 `train_dsg_alpha`、`train_slqc_alpha`（每轮均值），TensorBoard写入 `Module/`。
通过这些量确认残差分支从零开始学习；不能只看模块配置为True就认定它有效。

以下工具只在CPU构造配置、优化器并加载COCO权重，记录7类头形状与额外参数；
不读训练图像，不做forward/optimizer step，不用CUDA，不是训练效果验证，也不设为训练启动前置条件：

```bash
PREP=$(mktemp -d /root/zxy/experiments/protocol_checks/sar_modules_cpu_XXXXXX)
python tools/inspect_sar_module_readiness.py \
  --checkpoint /root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth --output "$PREP"
```

默认检查DSG、SLQC与完整SCS+SLQC+DSG三种配置的实际工厂构造，独立/组合参数量分别为
19,537,922 / 19,561,667 / 19,659,040（trainable）。其余7项也通过解析和模块功能验证。

## Git 更新与环境

在服务器执行；保存项目的未提交修改，不 reset，不强制 checkout：

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
```

GitHub HTTPS 不通时，使用已传输 Git bundle 的 `git fetch /path/to/bundle HEAD`
与 `git merge --ff-only FETCH_HEAD`。本地未提交修改不会自动存在于服务器。

## 新训练：一次双卡条件启动

下面选择 DSG 主实验；切换成矩阵中的其他 YAML 即对应独立运行。

```bash
CONFIG=configs/deim_dfine/sar_aircraft_modules/dsg.yml
bash tools/run_sar_module_experiment.sh "$CONFIG"
```

启动器在临近 nohup 前只检查一次 GPU0/1：都无计算进程、utilization=0、显存≤256MiB 才启动。
任一卡忙就写 `NOT_STARTED.txt` 并退出；不排队、不等待后自动启动、不停止其他进程，也不使用2/3卡。
新训练总是 COCO `-t` 初始化，新建优化器，不接着 baseline 或 SCS 的最佳权重训练。

启动器会打印 `STARTED OUT=... PID=... LOG=...`，并保存：

- `resolved_config.yml`、`source_config.yml`、`config_path.txt`；
- `git_revision.txt`、`git_worktree_status.txt`、`run_metadata.json`；
- `dataset_stats.json`、`method_and_protocol.md`、`gpu_gate.json`；
- `start_command.sh`、`nofile_soft.txt`、`launcher.pid`、`console.log`。

metadata 记录模块开关、seed、全局batch、输入/增强、优化器和调度、EMA/AMP、初始化及验证协议。
脚本执行 `nohup ... > console.log 2>&1 < /dev/null &`，先保存 `$!`，再 `disown`。

## 日志、进程、指标与 TensorBoard

每次重连均激活环境并重新设置实际 `OUT`：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
cd /root/zxy/deim
OUT=/root/zxy/experiments/替换为启动器打印的实际新目录
export OUT
tail -f "$OUT/console.log"
```

另开终端查看：

```bash
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
watch -n 2 nvidia-smi
tail -n 3 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

检测的 JSONL 在 `log.txt`；`test_coco_eval_bbox` 实际是 val 指标，0/1/2 为AP/AP50/AP75，8为AR100，
原值乘100报告百分数。SCS组合还记录其两项损失与描述有效率。
TensorBoard 在 `summary/`，COCO评估缓存为 `eval/latest.pth`。

汇总当前最佳逻辑epoch（不是直接看检查点内部last_epoch）：

```bash
python - <<'PY'
import json, os, pathlib
rows = [json.loads(line) for line in (pathlib.Path(os.environ['OUT'])/'log.txt').read_text().splitlines() if line.strip()]
best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
print('completed_epochs:', len(rows), 'best_logical_epoch:', best['epoch'])
print(dict(zip(['AP','AP50','AP75'], [100*x for x in best['test_coco_eval_bbox'][:3]])))
PY
```

## 恢复已停止的中断运行

先确认原运行已停止；不要对活跃任务再次恢复。仍须GPU0/1同时空闲。
恢复使用原目录的 `resolved_config.yml` 和 `last.pth`，不读当前新配置、不同时提供 `-t`。

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
cd /root/zxy/deim
OUT=/root/zxy/experiments/替换为已停止的实际运行目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复日志以 `resume_日期时间.console.log` 单独保存，metadata、GPU检查、命令和PID也带本次恢复时间。
原新训练的 metadata/config/console 不覆盖，`launcher.pid` 更新为最新 launcher。
当前 solver 只在强增强停止前更新 `last.pth`，正常最后为epoch89；它不是最终或最佳权重。
收尾会回载第一阶段最佳权重，因此检查点内部last_epoch可能回退；按实际验证AP比较best_stg1/2。

## 重评验证集

训练完成后或已停止时，确认双卡空闲再执行。`BEST` 填实际验证AP胜出的检查点，不能仅凭阶段名选择。
评估保存到新目录，不覆盖训练日志：

```bash
BEST=best_stg1.pth
EVAL=$(mktemp -d "$OUT/val_recheck_XXXXXX")
ulimit -Sn 65535
nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/$BEST" --test-only --use-amp \
  --output-dir "$EVAL" --summary-dir "$EVAL/summary" > "$EVAL/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$EVAL/launcher.pid"
disown "$PID"
tail -f "$EVAL/console.log"
```

## 验证范围与后续评价

只验证变换重建、局部坐标/布局、梯度、初始化恒等、DN对应、AMP精度和配置协议这些改动相关功能。
不做文件哈希校验，不把额外启动前试跑设为交付前置条件。本次没有完整训练结果，不报告AP收益。
后续比较除AP/AP75外，继续检查同框多类、相同Recall下背景FP/图，以及共同检出GT的宽高误差。
候选方向门控不能单凭热图命名成真实物理去斑，水平框回归也不等于目标轮廓分割。
