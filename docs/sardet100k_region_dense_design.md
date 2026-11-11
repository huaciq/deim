# RegionDense：保留目标尺度与局部背景的 Dense O2O 候选方案

状态：36 轮训练与完整 test 评估已完成。val AP 66.12、test AP 63.43；相对 E008 分别 +0.26、+0.11，未达预设晋级线。
详细证据见 [RegionDense 结果](sardet100k_region_dense_results.md)。以下保留实验前设计及判断标准。

## 1. 具体解决什么

原 Dense O2O 通过拼图和 MixUp 增加每张输入中的真实目标，从而增加一对一匹配的正样本。
它不需要对一个 GT 匹配多个 query。保留这个机制，不必保留原来的整图缩小拼接方式。

SARDet 诊断发现原 Mosaic 会增加贴近像素极限的目标，并改变部分目标邻域。
E008 去掉 Mosaic 与 MixUp 后优于 E000，但这不能证明所有差异都由缩小造成：bridge 的严格定位覆盖改善，tank 的部分定位覆盖反而下降、排名错误减少。
因此，本方案解决可直接验证的数据构造问题，不把尚未成立的单一因果解释当结论。

普通缩放后 12 像素的目标，进入 RegionDense 仍是 12 像素；在目标周围保留原图背景，再把这些区域拼到一张 640×640 图里。
额外目标仍按原 Hungarian 匹配、原框回归与原 MAL 学习。推理模型完全相同。

## 2. 固定的构造规则

1. 先按普通训练路径将源图缩放到 640×640，不再为拼图额外缩小。后续 batch 多尺度与 E008 相同，仍可能把输入缩到 480；这里保证的是“不引入额外缩小”。
2. 每个 GT 向四周扩展一个自身长边，作为需要保留的局部背景范围；到原图边界为止。这个范围是明确的工程假设，不声称覆盖全部 SAR 阴影或成像上下文，也不能修复原图已有截断。
3. 主图全部目标及上述邻域都能放进 320×320，则选四格；能放进 320×640 或 640×320，则选两格；否则用原整图。主图不会为了凑拼图而删目标。
4. 补图从预先建立的训练区域库中选择。可以裁取补图中的局部，但所有与裁块相交的 GT 都必须完整保留，且其局部背景范围也完整保留；不能留下未标注的半个目标。
5. 区域先按源图的主类别分桶，每桶先均匀选源图，再选该图的有效裁块。按训练集的 GT 类别比例计算各桶抽样概率，校正“可裁区域大多来自 ship”的偏差。对包含多个类别的区域，计算所有可见 GT 的类别向量，不只算桶名。
6. 同一拼图不重复使用同一源图；总目标数超过 300 时回到整图，不丢 GT 去适配 query 数。
7. 仅在 epoch index [4,20) 以 0.5 概率尝试，其他轮次走普通路径。MixUp 关闭；MAL、模型、初始化、学习率、EMA、AMP、batch size 与 E008 相同。

类别概率只由训练标注求得。设桶 j 的平均目标向量是 m_j，训练集各类目标比例是 q，先非负求解 Mᵀw=q，再归一化 w。
这使新增目标的类别比例在期望上等于训练集；避免重复源图与 300 目标上限会带来很小的抽样偏差，实际曝光量由 CPU replay 单独报告。
它不保证每个实例被补入的次数相同，也不保证类别内部的尺寸分布完全不变。

局部邻域外仍有拼接缝和跨来源背景。方法减少目标附近的这种变化，并没有消除整幅图的分布差异。

## 3. CPU 证据及其边界

2026-11-10 在服务器训练标注上完成两轮几何设计检查，均不使用 val/test 结果选择参数：

| 构造方式 | 普通输入平均 GT | 每次尝试组图后的平均 GT，含整图回退 | 类别曝光变化 |
|---|---:|---:|---|
| 只选能容纳全部 GT 的补图 | 2.10 | 3.99 | ship 2.23 倍，car 1.05 倍，偏差明显 |
| 允许完整局部区域补图，并校正类别比例 | 2.14 | 4.40 | 六类约 2.01～2.10 倍 |

以上第二行来自 10,000 次几何抽样，**是每次都尝试拼图时的结果，不是整个训练的平均密度**。
配置概率是 0.5，对应增强阶段预期约 3.27 GT/输入，比 2.14 增加约 53%。
约 66% 主图可用两格或四格；加上概率分支，增强阶段约 67% 输入仍为普通整图。
原始诊断：`/root/zxy/experiments/diagnostics/region_dense_design_20261110_b01/`。
正式实现的区域库和真实像素核验分别通过下列命令生成，应以它们的报告为实施依据。

### 正式实现的实际结果

代码提交 `b75cbaa` 已在新服务器完成以下工作（没有运行模型训练）：

- 区域库覆盖 94,493 张训练图、198,676 个 GT；构建耗时约 92 秒，只读取训练标注。
- 全训练集主图分流：42,718 张四格、20,202 张两格、31,573 张整图。每种分流都保留主图全部 GT。
- 正式数据构造器重放 10,000 次：普通输入平均 2.1286 GT；每次尝试组图并计入回退时 4.4100 GT；按配置 0.5 概率折算，增强阶段平均 3.2693 GT，增加 53.6%。
- 这 10,000 次中，按概率折算后的普通整图比例约 66.4%；其余为两格或四格。
- 主图普通抽样不变。新增 GT 类别比例在设计期望上匹配训练集；这次重放的六类曝光倍率，在真实 0.5 概率下分别为 ship 1.550、aircraft 1.494、car 1.540、tank 1.564、bridge 1.553、harbor 1.429。小类别有有限样本波动，不能把期望相同写成每次重放精确相同。
- 200 张真实组图核验：裁块像素逐值等于普通 640 缩放后的对应区域；GT 边长最大数值差约 1.14e-13 px；后续 Sanitize 删除目标数为 0；完整转换管线输出有效。
- 全部 623,938 个补图候选区域，按实际数据集的 float32 框转换重新检查：没有出现可见目标漏标、裁断受保护邻域或类别计数不一致。
- Windows `pytorch` 与服务器 `deim_zxy` 均通过 5 项针对性功能测试。

注意，“增加 53.6%”只针对启用增强的 16 轮；按照相同抽样均值折算整个 36 轮，平均约 2.636 GT/输入，比普通训练增加约 23.8%。这不是监督质量或 AP 增益。

实际产物：

```text
/root/zxy/experiments/region_banks/sardet100k_train_region_dense_v1.npz
/root/zxy/experiments/region_banks/sardet100k_train_region_dense_v1.json
/root/zxy/experiments/diagnostics/region_dense_design_20261110_b01/implementation_audit/report.json
/root/zxy/experiments/diagnostics/region_dense_design_20261110_b01/implementation_audit/example_003_31494.jpg
/root/zxy/experiments/diagnostics/region_dense_design_20261110_b01/float32_geometry_check.json
```

示例中 image 31494 的主图有 6 个 tank；组图仍保留这 6 个目标及原有尺度，并补入 3 个完整目标，总数为 9。另一张 bridge 主图通过两格拼接从 2 个目标增至 17 个，展示两种布局均能工作。
这些展示不是模型预测图，绿色框是主图 GT，橙色框是补图 GT。

历史准备状态：2026-11-10 区域库验证完成时 GPU 4、5 被占用，未启动。随后依用户指令正式启动，并于 2026-11-11 完成训练及 test 评估；结果见文首链接。

本地 `tests/test_region_dense.py` 检查：四格/两格/整图分流、主图 GT 保留、所有可见 GT 邻域保留、裁块像素与普通缩放结果一致、框边长不变、类别期望比例、增强轮次及完整数据管线；同时比较解析后的配置，确认其余训练协议与 E008 相同。
这些检查不证明模型能从新增目标中学到更多，也不证明训练更快。

## 4. 唯一一次首轮训练如何判断

- 主对照为新服务器的 E008，36 轮 val AP=65.8618；E000=65.2943 为原始 DEIM 对照。
- 首要晋级线：同 36 轮协议 val AP 至少比 E008 高 0.5，即约 66.36。只超过 E000、仍低于 E008，不算 Dense O2O 改造有效。
- 同时检查 AP75、APs，以及 tank/bridge 的 <16 px 组；分别报告，不能用总体 AP 掩盖明显的关键分组退化。
- 记录达到 60/65 AP 的实际耗时和逐轮曲线。额外读取区域可能增加数据耗时；每轮更慢也可能更早达到目标 AP，反之亦然。
- 先用 val 判断，test 留作确定方案后的报告；不依据 test 挑区域参数。
- 36 轮属于固定预算比较。现有曲线不能证明两组已经充分收敛；候选通过后才值得给候选与对照共同延长协议。不能只延长候选后宣称胜过 36 轮对照。
- 若连 E008 都不能超过，不继续围绕概率、边距进行一串微调，应回看真实匹配/排名收益是否存在。

已有 Select-Mosaic 等研究提出区域选择式拼图，因此“先裁区域再拼”本身不能声称是首次提出。
潜在研究贡献是 SAR 条件下对密集监督有效性的诊断，以及保尺度、保邻域、保主图目标的构造与实验效果。最终新颖性和效果必须由文献比较及训练证据支撑。
参考：[DEIM](https://arxiv.org/html/2412.04234v3)、[Select-Mosaic](https://arxiv.org/abs/2406.05412)。

## 5. 文件与可复现性

- `engine/data/region_bank.py`：几何规则、类别抽样概率、组图计划。
- `engine/data/transforms/region_dense.py`：训练数据变换；注册于 transforms/__init__.py。
- `tools/build_region_dense_bank.py`：只读取 train 标注生成 NPZ 区域库与 JSON 统计。
- `tools/audit_region_dense.py`：CPU replay、真实像素与坐标核验、示例图片。
- `configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_36e_b32.yml`：单独配置，继承 E008。
- `tests/test_region_dense.py`：对应功能验证，不运行模型训练。

固定：seed 0、global batch 32（双卡每卡 16）、基准输入 640、batch 多尺度 480～800 到 index 32、36 轮、flat 20、warmup 2000、LR 4e-4 / backbone 4e-5、EMA 0.9999、FP16 + FP32 loss/retry、DEIM-D-FINE-M COCO 90e 初始化、原 MAL、val COCO bbox AP。
记录 Git revision；区域库另外保存构造 seed 20261110、边距 1.0、搜索预算 16 个 anchor/图/布局、至多 8 个区域/图/布局。
区域库、图片、训练输出均不提交 Git。

## 6. 新服务器交接命令

本次用户指定新服务器及 GPU 4,5，故这里使用新服务器路径。先查看 git status；若存在阻碍同步的改动或分叉，保留并解决，不能强制覆盖。

```bash
cd /root/zxy/deim
conda activate deim_zxy
git status --short --branch
git fetch origin
git switch main
git -c pull.rebase=false pull --ff-only origin main
```

本次服务器访问 GitHub 超时，已通过 Git bundle 快进到与 origin/main 已发布的同一代码提交；没有直接拷贝文件覆盖工作树。若服务器显示相对旧 origin/main “ahead”，这是远程跟踪引用尚未刷新；正常网络恢复后运行上面的 fetch/pull 即可。离线 Git 交接方法如下，路径需使用实际传输的 bundle：

```bash
git status --short --branch
git fetch /root/zxy/experiments/diagnostics/region_dense_design_20261110_b01/source.bundle main
git merge --ff-only FETCH_HEAD
```

只需一次的训练区域库准备（已存在同一库时直接复用；需要新参数则换新文件名及新配置，不覆盖）：

```bash
export BANK=/root/zxy/experiments/region_banks/sardet100k_train_region_dense_v1.npz
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/build_region_dense_bank.py \
  --annotations /root/zxy/TCRA-SAR/datasets/SARDet_100K/Annotations/train.json \
  --output "$BANK"
export AUDIT=/root/zxy/experiments/diagnostics/region_dense_impl_$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/audit_region_dense.py \
  --config configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_36e_b32.yml \
  --output "$AUDIT" --draws 10000 --pixel-checks 200
```

正式启动，使用现有 launcher 创建独立输出目录、保存解析配置、协议、Git revision 和全套参数；内部使用 `nohup ... > console.log 2>&1 < /dev/null &`，立即保存 `$!` 到 launcher.pid 后 disown。它只检查一次 GPU 是否空闲，不排队、不停止其他任务。

```bash
cd /root/zxy/deim
conda activate deim_zxy
GPU_IDS=4,5 \
EXPERIMENT_ROOT=/root/zxy/experiments \
INITIAL_WEIGHTS=/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth \
RUN_PREFIX=deim_m_sardet100k_region_dense \
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_36e_b32.yml
```

启动成功后把打印的 OUT 绝对目录粘贴到下面；重连时重新设置 OUT 并激活环境。

```bash
conda activate deim_zxy
export OUT=/root/zxy/experiments/替换为启动时打印的完整目录
tail -f "$OUT/console.log"
nvidia-smi -i 4,5
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
tail -n 5 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 127.0.0.1 --port 6006
```

指标与最优 epoch（log.txt 数值乘 100）：

```bash
python - <<'PY'
import json, os
from pathlib import Path
rows = [json.loads(line) for line in (Path(os.environ['OUT'])/'log.txt').read_text().splitlines() if line.strip()]
for row in rows[-5:]:
    a = row['test_coco_eval_bbox']
    print(row['epoch'] + 1, dict(AP=100*a[0], AP50=100*a[1], AP75=100*a[2], APs=100*a[3]))
best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
stage = 'best_stg2.pth' if best['epoch'] >= 32 else 'best_stg1.pth'
print('Best:', best['epoch'] + 1, 100*best['test_coco_eval_bbox'][0], stage)
PY
```

确认原训练已经停止后恢复（不用同时传 -t 和 -r）：

```bash
cd /root/zxy/deim
conda activate deim_zxy
export OUT=/root/zxy/experiments/替换为要恢复的完整目录
GPU_IDS=4,5 bash tools/run_sar_module_experiment.sh --resume "$OUT"
tail -f "$OUT"/resume_*.console.log
```

launcher 从 last.pth 恢复。当前 solver 的 last.pth 只更新到 no-aug 阶段之前，**不代表第 36 轮**；若在最后 4 轮中断，这条命令会从阶段边界重跑。final.pth 是当前代码保存的计划末轮状态；选择最佳模型仍以 val AP 为准。

对实际 val AP 最好的 checkpoint 做独立验证，BEST 用上面指标脚本输出的文件名：

```bash
export BEST=best_stg2.pth  # 按实际指标选择，也可能是 best_stg1.pth
export EVAL="$OUT/eval_selected_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL"
CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/$BEST" --test-only \
  --output-dir "$EVAL" > "$EVAL/console.log" 2>&1
```

TensorBoard 若从本机访问，另建 SSH 本地转发 `ssh -p 20106 -L 6006:127.0.0.1:6006 root@10.11.154.192`，浏览器打开 localhost:6006。
