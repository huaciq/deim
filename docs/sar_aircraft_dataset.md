# SAR-AIRcraft-1.0 数据准备与 DEIM 使用

源压缩包：`/root/zxy/datasets/SAR-AIRcraft-1.0.zip`。
数据根目录：`/root/zxy/datasets/SAR-AIRcraft-1.0`。

准备工具 `tools/dataset/prepare_sar_aircraft.py` 去掉压缩包的 origin_data/ 外层目录，
保留 JPEG、XML、ImageSets 原文件，再额外生成 COCO 标注；不修改图片尺寸或原始标注。
本次只准备数据和配置，不启动训练，不停止现有 SARDet-100K 训练。

## 数据与转换口径

原包包含 4,368 张 JPEG、4,368 份 XML、4 份划分清单，共 16,463 个目标。
类别为 A220、A320/321、A330、ARJ21、Boeing737、Boeing787、other。
这些总量与[发布方数据介绍](https://www.radars.ac.cn/web/data/getData?newsColumnId=81053ef7-8327-4f09-88c0-ea061f64f9ba&pageType=en)一致。

- 原始 train=3047、val=442、test=879；三者不重叠；trainval=3489，即 train+val。
  保留这四份列表，不重新随机划分。
- 默认训练配置使用 train.json 训练、val.json 选择模型。test.json 单独用于最终测试，
  trainval.json 同时提供，供今后明确决定合并训练时使用。
- 分类保持原始七类，ID 按类名排序映射为 0..6，remap_mscoco_category=False；不合并为单类。
- XML 端点直接转为 `[xmin,ymin,xmax-xmin,ymax-ymin]`，不加减 1、不平移、不裁剪。
  本仓库 VOCDetection 也直接使用 XML 端点，此转换与其几何口径一致；不额外推断或修改
  原数据的坐标起点约定。COCO 中保留 source_voc_bbox，便于还原对应原始框。
- 比对每张图的 JPEG 头部尺寸和 XML，核对路径、框范围、类别及原始划分关系。
  正常转换的 source bbox 往返误差应为 0，不进行文件哈希检查。
- 源数据所有 difficult=0；保留 source_difficult/source_truncated，iscrowd=0。

## 服务器更新与准备

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
python tools/dataset/prepare_sar_aircraft.py \
  --archive /root/zxy/datasets/SAR-AIRcraft-1.0.zip \
  --dataset-root /root/zxy/datasets/SAR-AIRcraft-1.0
git rev-parse HEAD > /root/zxy/datasets/SAR-AIRcraft-1.0/preparation_git_revision.txt
```

GitHub HTTPS 连接失败时，从开发端通过 Git bundle 传输提交，服务器执行
`git fetch /tmp/本次.bundle main`、`git merge --ff-only FETCH_HEAD`，保留无关工作区文件。
数据仅在服务器保存，不提交 Git。解压目标应为新目录；已解压后不再传 --archive。
若需从原始文件重新派生标注，使用新的 --output-dir，保留已有 COCO 文件。

完成后的结构：

```text
/root/zxy/datasets/SAR-AIRcraft-1.0/
├── JPEGImages/                 # 4368 张原图
├── Annotations/                # 4368 份原 XML
├── ImageSets/Main/             # 原 train/val/trainval/test.txt
└── COCO/
    ├── train.json
    ├── val.json
    ├── trainval.json
    ├── test.json
    ├── category_map.json
    └── stats.json
```

所有 COCO image.file_name 使用 `JPEGImages/0000001.jpg` 这种根目录相对路径，
因此 img_folder 指向 **SAR-AIRcraft-1.0 根目录**，不能再额外指向 JPEGImages。

## DEIM 固定协议配置（仅交接，当前不启动）

配置：`configs/deim_dfine/deim_hgnetv2_s_sar_aircraft_120e_b32_paper_nozoom_nocrop.yml`。
继承现有固定协议，覆盖类别数、数据路径、输出目录：120 轮、global batch 32、GPU 0/1、
640 基础输入、主/骨干 LR=4e-4/2e-4、warmup=2000、flat=64、stop=118、
Mosaic/MixUp p=0.5、ZoomOut/IoUCrop 关闭、EMA/AMP/原 FP32 异常重算，COCO 初始化。
该小数据集按固定数字仍使用 2000 步预热，没有自动重调学习率或缩短预热。

未来用户明确要求训练时，从 COCO 初始化并使用新目录：

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
CONFIG=configs/deim_dfine/deim_hgnetv2_s_sar_aircraft_120e_b32_paper_nozoom_nocrop.yml
OUT=$(mktemp -d /root/zxy/experiments/deim_s_sar_aircraft_paper120_b32_seed0_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_XXXXXX)
export CONFIG OUT
git rev-parse HEAD > "$OUT/git_revision.txt"
python - <<'PY'
import json,os,pathlib,yaml
from engine.core.yaml_utils import load_config
p=pathlib.Path(os.environ['OUT']);c=load_config(os.environ['CONFIG'],{})
c.pop('__include__',None);c.update(output_dir=str(p),summary_dir=str(p/'summary'),seed=0)
(p/'resolved_config.yml').write_text(yaml.safe_dump(c,sort_keys=False))
(p/'run_metadata.json').write_text(json.dumps(dict(git_revision=(p/'git_revision.txt').read_text().strip(),seed=0,devices=[0,1],global_batch=32,initialization='/root/zxy/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth',config=c),indent=2))
PY
cp /root/zxy/datasets/SAR-AIRcraft-1.0/COCO/stats.json "$OUT/dataset_stats.json"
ulimit -Sn 65535
nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -t /root/zxy/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --seed 0 --use-amp --output-dir "$OUT" \
  > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
tail -f "$OUT/console.log"
```

今后中断时恢复（确认原进程已经停止，-r 和 -t 不同时使用）：

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/替换为需要恢复的实际运行
LOG="$OUT/resume_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).console.log"
ulimit -Sn 65535
nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/last.pth" --seed 0 --use-amp --output-dir "$OUT" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
```

last.pth 只在 epoch<118 更新，正常最后为 epoch 117，不是最终权重。
查看日志/指标及 GPU；重连时明确 OUT 并激活 deim_zxy：

```bash
tail -f "$OUT/console.log"
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
watch -n 2 nvidia-smi
tail -n 3 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

log.txt 的 COCO 指标 0..1，AP/AP50/AP75 为 test_coco_eval_bbox 的前 3 项，乘 100 报告。
训练逐轮验证 val.json。按验证 AP 选 best_stg1/best_stg2 后，单独评估保留的 test.json：

```bash
STAGE=best_stg1  # 根据实际验证 AP 选择，不能仅凭阶段名
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
