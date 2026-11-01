# REM 102 轮：分类 logits、LQE 与子带残差机制诊断

## 本轮要回答的问题

首轮诊断复算 AP64.5077→65.2237，A330 AP+5.1401、A220 AP-1.9668。
A220在score>=0.3时两组TP/FP均213/66，但误检平均分数0.5805→0.6506；
共同检出的167个A220中目标平均IoU反而提高约0.00215。
下一步区分：错误分数变化来自原始分类logits、LQE几何加分，还是特定残差分支的影响。

本轮固定已有最佳权重，全部为推理、读出和CPU统计，不重新训练。
不修改生产模型、原配置、原标注或任何checkpoint。原训练复现/恢复仍见
[既有交接](sar_rem_per_band_experiment.md)，本轮不启动该训练。

## 五个固定版本

| 版本 | checkpoint | LL倍率 | LH/HL/HH倍率 |
|---|---|---:|---:|
| baseline | 原baseline最佳 | 不存在REM | 不存在REM |
| rem | 原REM最佳 | 1 | 1 |
| rem_half | 同一个REM最佳 | 0.5 | 0.5 |
| rem_ll_half | 同一个REM最佳 | 0.5 | 1 |
| rem_hf_half | 同一个REM最佳 | 1 | 0.5 |

后三个版本通过临时forward hook缩放卷积分支输出，仍只使用一次DWT/IDWT。
原共享alpha与全部参数不变；诊断钩子在结束后移除。普通rem版本不改分支输出。
这些结果是已训练网络上的受控推理干预，不能写成对应结构重新训练后的消融结果。
统一使用FP32、固定640、EMA（若原checkpoint保存）、原top300后处理，无额外NMS或置信度预过滤。
每次都用GPU0、1，依次完成五次全验证集推理；它们不是五轮或五组训练。

## 读出与比较定义

- 从实际eval Decoder层的LQE入口读取7类raw logits，从reg_conf输出读取实际LQE标量。
- 保存每个图像的全部Query，而非只保存最终top300，以便本地移除LQE并重新选择top-K。
- 保存原始分类margin、raw top-1与真实类别关系，区分真正的类别犹豫和LQE统一加分。
- 保存投影输入、实际总残差、LL/LH/HL/HH分量的通道平均绝对响应。
- 分量绝对值通过Haar合成的符号/倍率解析计算，不额外调用IDWT。
- 内部3x3采样位于0.75框；外环12点在1.05框四边。采样仅用于诊断，不参与预测或训练；
  超出有效特征中心范围的点被排除，另保存有效比例，避免图像外零填充产生虚假对比。

特征响应不是SAR物理散射功率，各分量绝对值之和也不等于总残差绝对值。
同一GT或同一固定背景参考区域用于跨模型/版本配对，绝不直接对齐Query编号。
对参考区域内达到指定IoU的全部Query，选择目标类别最终分数最高者读取raw/LQE；
同时保存候选是否存在。配对均值只计算两边都存在候选的对象，缺失不按logit=0处理。

四组固定观察对象来自首轮两组预测的并集：

- a220_true：全部247个A220 GT，类别0、几何IoU至少0.5。
- a330_true：全部27个A330 GT，类别2、几何IoU至少0.75。
- a220_wrongclass：在首轮任一模型中被高分误判为A220的其他类GT，按GT ID合并。
- a220_background：首轮任一模型的A220背景误检区域，按同图IoU合并参考区域。

对用户已下载的首轮真实预测生成观察对象，得到247个A220真值、27个A330真值、
58个A220错类GT参考对象和24个背景参考区域，共356条观察记录。

错误定义均相对于现有标注。groups可能共享同一GT的不同类别观察，不是互斥的全数据集划分。
报告同时给出五版本完整COCO指标，避免只分析人为挑选的失败样本。

## 服务器更新与启动

复用已经完成的首轮诊断目录，不需要再搬baseline文件或标注。
先检查工作树并保留本地改动，Git更新采用ff-only。GPU0、1可用于推理后执行：

```bash
cd /home/zxy/sar/repos/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export COMPARISON=/data2/zxy/sar/experiments/diagnostics/rem102_comparison_20261101_173951_Kpvr6X
nvidia-smi -i 0,1
bash tools/run_rem_mechanism_probe.sh "$COMPARISON"
```

启动器新建/data2/zxy/sar/experiments/diagnostics/rem102_mechanism_*目录，
nohup后台执行，stdout/stderr写console.log，stdin来自/dev/null；先保存$!到launcher.pid再disown。
工作进程为每个版本依次调用torchrun --standalone --nproc_per_node=2。
不停止已有任务、不等待空卡或排队。STARTED表示工作进程启动，EXPORT_DONE表示对应版本导出成功。

## 看进度、报告和下载结果

把OUT替换为新打印的rem102_mechanism目录。重连先激活环境并重新设置OUT：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/diagnostics/替换为本次rem102_mechanism目录
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
nvidia-smi -i 0,1
tail -f "$OUT/console.log"
```

某版本的细节日志在对应目录，例如：

```bash
tail -n 20 "$OUT/rem/export.console.log"
tail -n 20 "$OUT/rem_hf_half/export.console.log"
```

全部完成并出现DONE后：

```bash
cat "$OUT/report/report.md"
ls -lh "$OUT/mechanism_bundle.tar.gz"
```

只下载mechanism_bundle.tar.gz。包中包含五版本的完整预测、全部Query读出、配置副本、
比较报告及少量原始案例图片，不含checkpoint。案例图片保持原样，框和类别在examples/cases.json中。
本轮不产生训练TensorBoard事件；推理、CPU分析进度看console.log和各版本export.console.log。

## 中断恢复

原诊断进程已结束后执行，复用export.done标记的已完成版本：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/diagnostics/替换为本次rem102_mechanism目录
bash tools/run_rem_mechanism_probe.sh --resume "$OUT"
```

恢复进度写resume_*.console.log。此处恢复的是诊断，不是训练，不使用-t或train.py -r。

## 结果文件与分析顺序

1. intervention_metrics.csv：全验证集AP/AP75/APm、A220与A330指标，比较固定分支减半的影响。
2. paired_summary.csv：相同对象的Δraw、ΔLQE、Δfinal、Δscore、margin与IoU；rem参照baseline，减半版本参照rem。
3. group_summary.csv：各组候选存在数量、原始top-1关系与响应幅度；先看配对数与缺失数再解释均值。
4. anchor_evidence.csv/json：逐目标完整读出、七类竞争的派生信息和实际选中Query编号。
5. 各版本queries.npz：image_ids/orig_sizes/boxes/raw_logits/quality/final_logits/core_abs/ring_abs/valid_fractions。
6. raw_only_evaluations：同一批已解码框移除LQE，使用全部Query重新做top-K；这是排序反事实，不是新模型性能。

先比较A220错类/背景组与真阳性组的分解，再判断原始分类项或LQE项哪一部分值得改进；
若分支干预改变同一错误的分数但保留A330收益，才继续研究对应的残差约束。
统计分解和推理干预均不直接证明训练中的因果机制，不按最高干预AP选择“正式结果”。

## 本地验证与限制

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_rem_mechanism_probe.py -v
```

9项通过：真实四层Decoder的变动LQE读出与logits分解、原预测/权重保持、Haar解析响应与单次变换、
无效采样排除、跨分片数组对齐、GT而非Query编号配对、缺失候选处理、移除LQE的排序反事实、
准备/报告/打包和中断恢复。未执行旧服务器真实权重的CUDA双卡推理；当前没有新增训练结果。

启动器及本文5段Bash命令语法验证通过。结果包归档首轮reference_summary.json，
在本机解压后可独立复算，无需访问旧服务器的原诊断目录。
