# SAR-AIRcraft：冻结检测器的局部证据排序探针

## 固定问题与设计

只检验局部图像证据是否提供现有分数、类别竞争和 FDR 统计之外的排序信息。
不训练检测器，不改框、类别、候选集合、NMS 或原 baseline 配置。
实际 AP 必须由已有 SAR-AIRcraft checkpoint 和原始验证集计算。

从原 HGNetv2 stage 1（S2 / stride 4）和 decoder 输入的第一个尺度
（P3 / stride 8）只读提取特征。每层用固定 seed 的随机正交投影压到 8 通道：
框内采样 3×3 网格；四边各取 8 个沿边采样点，保留法向 -5%、0、+5%
框宽/高三个位置的沿边均值与标准差。另记录采样点是否在图内。
采样使用预测框，绝不使用 GT 决定 ROI。局部描述用 FP16 缓存、读回 FP32 拟合；
检测框和原分数始终保留 FP32。该探针测试固定描述能否利用信息，
不预先安装 LSNet、Strip 或额外可训练特征提取模块；阴性结果不排除所有局部模型。

固定比较：score、cls、cls_fdr、cls_fdr_roi、cls_fdr_border、cls_fdr_local。
所有拟合器是相同隐藏层宽度 64→32 的两输出 MLP，输入维度及参数量单独记录，
不声称严格等参数。比较原分数和免拟合类别竞争规则
`s * sqrt(s / sum(all_class_sigmoid_scores))`。

两个拟合目标是同类别最大 IoU 与忽略类别的最大 IoU。
它们不是 COCO 一对一 TP 标签，也不学习重复检测抑制；最终评价始终使用原 COCO
匹配协议。额外在几何 IoU≥0.5、最近 GT 类别正确的预测上，按类别和原分数区间
报告几何质量的 Spearman 相关，避免把类别可信度改善直接称为定位信息改善。

默认只读官方 val，2 折、3 次重复，seed 0/1/2。优先使用用户提供的
`image_id -> scene_id` JSON；否则用 32×32 灰度缩略图 Pearson 相关≥0.95
的连通分量分组，不使用标签或文件哈希。它只是近重复分组，不能保证隔离所有同场景图。
同组所有图及其全部候选始终在同一折。归一化只在训练折拟合。

每个拟合器固定训练 60 epochs，AdamW lr=1e-3、weight_decay=1e-3，
batch=4096、dropout=0.2、分数加权 soft BCE；不通过验证 AP 选 epoch。
两个 GPU 并行导出不同图像、随后并行执行不同折的拟合任务，不对同一个拟合器做 DDP。
检测器 FP32 / eval / no_grad，优先读取 checkpoint 中的 EMA；拟合器 FP32、无 EMA。
导出 global batch=16（双卡各 8），不会把训练 baseline 的全局 batch 再乘 GPU 数量。

候选信号：local 的平均同类别目标 AP 比原分数、固定竞争规则、score/cls/cls_fdr
中最强的已有信号对照高≥1.0，且相对 cls_fdr 在三次重复中均为正。
几何解释另外依赖几何目标的 AP 与条件相关。
重复折不是独立 detector seeds，标准差不是置信区间。test 不参与拟合或参数选择。
原检测器 checkpoint 曾由 val 选择，所以这次 val 交叉拟合也是探索性诊断；
它不提供新的独立测试结果，也不解决原 train–val 场景交叠。

## 文件与运行产物

- `tools/local_evidence_probe.py`：只读导出、分组、拟合、COCO 评价与恢复。
- `tools/run_local_evidence_probe.sh`：双卡 nohup 启动器，保存 PID 后 disown。
- `tests/test_local_evidence_probe.py`：本探针的功能验证。
- 不改动既有未跟踪的 `tools/quality_ranking_probe.py` 和
  `docs/research_diagnosis_2026-10-28.md`，也不把它们包含在本次提交中。

输出目录包含：

| 路径 | 内容 |
|---|---|
| `manifest.json`、`resolved_config.yml` | Git revision、工作区状态、软件版本、checkpoint、原检测训练协议和本次全部探针参数 |
| `features/<image_id>.npz` | 固定候选、分数、类别、框、FDR 和局部描述；按完整图保存 |
| `folds.json` | 场景分组来源、组数和每次重复的图像折分配 |
| `reference_metrics.json` | 拟合前打印并保存的原分数、竞争规则、两个 oracle 指标 |
| `models/<method>_repR_foldK.pth` | 最近完成 epoch 的拟合器、优化器、训练折归一化及 RNG 状态 |
| `models/<method>_repR_foldK.jsonl` | 每折每 epoch 的训练损失；不是检测器的 `log.txt` |
| `oof/*.npz` | 每个验证折的预测及原候选行号 |
| `ranks/*.npz` | 每个方法/重复的完整排序结果，同时保留原框、类别、分数 |
| `results.json`、`results.md` | AP/AP50/AP75、重复标准差、参数量、配对增益及几何条件相关 |
| `summary/` | TensorBoard：拟合损失与各重复的验证指标 |
| `console.log`、`launcher.pid` | 首次运行控制台和 torchrun launcher PID |

恢复时跳过已经完整导出的图像和已经完成的折。未完成的折从其最近完成的 epoch
继续；保存原始 CPU/CUDA RNG 和优化器状态。不把任意未完成批次伪装成已完成 epoch。
`resume` 直接读取原 manifest 的配置和数据路径；原 checkpoint、标注和图片应保持原样。

## Linux 更新与启动

分支：`codex/local-evidence-ranking-probe`。先检查服务器工作区；若有相关本地修改
或分歧，先查看差异，不 reset、不强制 checkout。这里通过 Git 更新源代码，不通过 SSH
传输，也不假设 Windows 修改自动出现在服务器。

```bash
cd /home/zxy/sar/repos/deim
git status --short --branch
git fetch origin
git switch codex/local-evidence-ranking-probe
git pull --ff-only origin codex/local-evidence-ranking-probe
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
```

这轮使用已记录 val AP 64.5104 的原 102e 多尺度 baseline，而非新 58e 实验。
记录中最佳 checkpoint 是该 run 的 `best_stg2.pth`，依据已有评价记录选取，
不是因为 stage2 文件名。原训练协议保持 global batch28、640 验证、480–800
训练多尺度、seed0、原数据增强/EMA/AMP和 COCO 初始化设置，全部复制进 manifest。
本次不重新训练检测器，不使用 `train.py -t`；只读取其训练完成的 checkpoint。

下面优先使用约定的 `/home/zxy/sar` 路径，若尚未迁移则使用诊断记录中的旧位置。
只读取这两个已知位置，不改动或准备标注。

```bash
BASE_NAME=deim_m_sar_aircraft_102e_b28_nozoom_nocrop_seed0_20261026_142643_LMw1Yy
BASE_RUN="/home/zxy/sar/experiments/$BASE_NAME"
if [[ ! -d "$BASE_RUN" ]]; then BASE_RUN="/root/zxy/experiments/$BASE_NAME"; fi
DATA_ROOT=/home/zxy/sar/datasets/SAR-AIRcraft-1.0
if [[ ! -d "$DATA_ROOT" ]]; then DATA_ROOT=/root/zxy/datasets/SAR-AIRcraft-1.0; fi
bash tools/run_local_evidence_probe.sh \
  "$BASE_RUN/resolved_config.yml" "$BASE_RUN/best_stg2.pth" \
  "$DATA_ROOT" "$DATA_ROOT/COCO/val.json"
```

有已有场景分组 JSON 时，作为第五个位置参数传给同一个启动器。例如末尾附上
`"/absolute/path/scene_groups.json"`。其格式为 `{"123": "scene_a", "124": "scene_a"}`，
覆盖所有 val image_id；禁止将标签或 GT 位置作为分组依据。

启动器内部正式命令使用 `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2`，
并以 `nohup ... > console.log 2>&1 < /dev/null &` 启动，先保存 `$!` 再 `disown`。
它不停止其他任务、不排队。输出 `STARTED OUT=...` 对应本次全新目录。

## 监控、评价与恢复

重连后明确设置 OUT 并激活环境：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
OUT=/home/zxy/sar/experiments/替换为STARTED输出的实际目录
tail -n 80 -f "$OUT/console.log"
```

另一终端监控和查看已有指标：

```bash
nvidia-smi -i 0,1
ps -fp "$(cat "$OUT/launcher.pid")"
cat "$OUT/reference_metrics.json"
tail -n 3 "$OUT"/models/*.jsonl
cat "$OUT/results.md"
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

`reference_metrics.json` 在导出后、拟合前产生；先看其 baseline 是否与原导出口径
64.51 AP 一致。数值不符时先检查 checkpoint、val 路径与配置，不能把其他基线混入比较。
`results.md` 在所有折完成后产生。原始 COCO stats 的 0/1/2 分别为 AP/AP50/AP75，
本工具已乘 100，不要再次乘。所有对照使用相同 top300 和 COCO maxDets100；
不加阈值筛选、NMS、框修正或类别改判。

若进程已经停止且需要继续：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
OUT=/home/zxy/sar/experiments/替换为原运行目录
bash tools/run_local_evidence_probe.sh --resume "$OUT"
```

恢复日志为 `resume_时间.console.log`，PID 更新到同一个 `launcher.pid`。
评价结果丢失或需重新汇总时，无需重跑导出和拟合：

```bash
python tools/local_evidence_probe.py report --output "$OUT"
```

读取关键差值和几何证据：

```bash
python - "$OUT/results.json" <<'PY'
import json
import sys
r = json.load(open(sys.argv[1]))
print(json.dumps(r['local_comparison'], indent=2))
print('original geometry correlation:', r['geometry_baseline'])
for method in ['cls_fdr', 'cls_fdr_roi', 'cls_fdr_border', 'cls_fdr_local']:
    p = r['probes'][method]
    print(method, 'same-class:', p['mean']['same_class'], 'geometry:', p['mean']['geometry'])
    print('geometry correlations:', [x['geometry_quality_correlation'] for x in p['repetitions']])
PY
```

只有同类别目标变好而几何目标/条件相关不改善时，优先解释为类别可信度或混合评分
改善；不能声称定位精度提升。即使排序成功，框本身也完全没有变。

## 本地验证与限制

Windows Conda `pytorch` / PyTorch 2.5.1 下，10 项定向功能测试覆盖：
四边采样方向和图外掩码；只读钩子及原后处理候选/坐标一致性；近重复和显式场景
分组不跨折；同类与纯几何目标/忽略区域；固定框重排的 COCO 指标；仅训练折归一化；
拟合器逐 epoch 恢复；几何条件相关；global batch/配置保存；验证折预测重组与输出。
另对 Bash 启动器运行语法检查。没有运行真实数据训练、哈希校验或冒烟测试。

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_local_evidence_probe.py -v
```

本地没有这次 baseline 权重、SAR-AIRcraft 标注和图像，也无法直连服务器，因此尚无
本方法的真实 AP、双卡运行耗时或显存实测。功能测试通过不等于局部证据有效。
