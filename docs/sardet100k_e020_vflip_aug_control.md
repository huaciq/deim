# E020：垂直翻转增强对照（旧服务器）

日期：2026-11-18。旧服务器 GPU 0、1（无法直连，经 Git 更新代码，由用户执行命令）。

## 要回答的问题

E013-v2（OCEF）把垂直翻转视图送进网络，在训练和推理时都与原图融合，相对 E000 val +1.97、test +2.35。
E018 已经排除了"第二次编码 + 门控融合"的通用效果（水平翻转视图只剩约 0.2–0.4 AP）。剩下最直接的质疑是：

> 既然有用的是垂直方向，直接在训练时加随机垂直翻转增强，不改网络，是否也能拿到这部分提升？

E020 只在 E000 的训练增强里加一个 `RandomVerticalFlip`（p=0.5），网络、推理和其余协议都不变。
D18 的裁块实验预测方向增强会损失方向线索、降低 AP；完整检测器上还没有验证过。

原 E012（垂直翻转 + 90° 旋转）同时改了两种方向，不能对应 OCEF 只用垂直翻转的设置，未运行；E020 是只含垂直翻转的版本，取代 E012。

## 为什么放在旧服务器、与谁比较

8 卡服务器另有安排。两台服务器上同一基线相差约 0.1–0.4 AP（E000 val 65.29 / test 62.39，E000-old val 65.15 / test 62.7），
所以 **E020 只与旧服务器上的 E000-old 比较**：

- E000-old：val AP 65.15，test AP 62.7，运行目录
  `/data2/zxy/sar/experiments/deim_m_sardet100k_deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop_seed0_20261107_083221_byTPNu`。
- 论文里按"相对同服务器基线的差值"报告：E013-v2 − E000 与 E020 − E000-old 并列。

## 配置与实现

- 配置：`configs/deim_dfine/deim_hgnetv2_m_sardet100k_vflip_aug_36e_b32_old_server.yml`，包含 E000-old 配置
  （`deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop_old_server.yml`）。YAML 列表不会合并，所以重写了整个增强列表：
  与 E000 完全相同，只在 `RandomHorizontalFlip` 之后插入 `RandomVerticalFlip`。垂直翻转在全部 36 轮都生效，
  与水平翻转一样不受 Mosaic 停用策略影响。
- 代码：`engine/data/transforms/_transforms.py` 注册 torchvision 的 `RandomVerticalFlip`（图像与框一起翻转）。其他配置不使用它，行为不变。
- 比较工具：`tools/sardet100k/compare_runs.py`，读取各运行的 `log.txt`（最高 val AP 那一轮）、val 的 `eval/latest.pth`
  （仅当最高 val 在最后一轮）和 test 的 `eval.pth`，打印总体与按类 AP 及相对第一个运行的差值，可选写 JSON。
- 定向测试 `tests/test_vflip_aug_e020.py`：配置相对 E000-old 只有增强列表不同，且只多出一个垂直翻转；
  变换已注册，图像与框按预期翻转（框 [1,2,3,5] 在高 10 的图上变为 [1,5,3,8]）；比较工具在合成评估数组上的读数正确。
  本地与 E018、E019 的测试共 11 项通过。

## 协议

与 E000-old 相同：SARDet-100K 官方 train / val（`/data2/zxy/sar/datasets/SARDet_100K`），seed 0，全局 batch 32（GPU 0、1 各 16），
COCO-M 初始化 `-t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth`，36 轮（flat 20，末 4 轮无 Mosaic），
AdamW 主 LR 4e-4 / 骨干 4e-5，EMA 0.9999，AMP，Mosaic / MixUp / Dense O2O 与 E000 相同，训练多尺度 480–800、评估 640，每轮完整 val。
按最高 val AP 选权重；test 在训练结束后评估一次，不参与选择。预计训练约 18 小时，显存与 E000 相同。

## 事先定好的判读

以 val AP 为主，记 r = (E020 − E000-old) / (E013-v2 − E000) = (E020 − 65.15) / 1.97，阈值与 E018 相同：

| 结果 | 判读 | 对论文的影响 |
|---|---|---|
| r ≤ 0.3（val ≤ 65.74） | 只做增强拿不到 OCEF 的提升 | 支持"保留原方向作参照、在网络内融合垂直视图"，E020 写入消融表 |
| 0.3 < r < 0.7 | 增强能拿到一部分 | 两者都报告；OCEF 的额外贡献按差值计 |
| r ≥ 0.7（val ≥ 66.53） | 只做增强就能拿到大部分提升 | OCEF 相对增强的优势不成立，需要重新评估主方法 |

同时看按类 AP：E013-v2 的增益集中在 bridge（val +7.31）和 harbor（+4.15）。若 E020 在这两类上也接近，同样说明增强能复现。
若 E020 低于 E000-old，与 D18 的预测一致（方向增强破坏方向线索）。单次运行，按上表直接读。

## 旧服务器命令

### 更新代码

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
git status --short --branch
git fetch origin
git switch main
git -c pull.rebase=false pull --ff-only origin main
git log --oneline -3
```

工作区有未提交修改或与 origin 分叉时先停下检查，不要 reset 或强制 checkout。日志第一行应是加入 E020 的提交。

### 启动前检查

```bash
python -m unittest tests.test_vflip_aug_e020 -v
ls /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
ls /data2/zxy/sar/datasets/SARDet_100K/Annotations/
nvidia-smi -i 0,1
```

### 启动

```bash
GPU_IDS=0,1 EXPERIMENT_ROOT=/data2/zxy/sar/experiments \
INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth \
RUN_PREFIX=deim_m_sardet100k bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sardet100k_vflip_aug_36e_b32_old_server.yml
```

脚本只在 GPU 0、1 都空闲时启动；新建运行目录，`nohup` 后台运行，stdin 来自 `/dev/null`，stdout/stderr 写入 `console.log`，
保存 `launcher.pid` 后 `disown`。最后一行打印 `STARTED OUT=... PID=... LOG=...`，把其中的 OUT 记下来：

```bash
export OUT=/data2/zxy/sar/experiments/<STARTED 行里的目录>
```

### 监控

重连后先重新设置 OUT 并激活环境：

```bash
cd /home/zxy/sar/repos/deim && conda activate deim_zxy && export OUT=/data2/zxy/sar/experiments/<运行目录>
```

```bash
tail -f "$OUT/console.log"
```

```bash
nvidia-smi -i 0,1; ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
```

```bash
python -c 'import json,os; [print(r["epoch"]+1, [round(100*x,2) for x in r["test_coco_eval_bbox"][:6]]) for r in map(json.loads, filter(str.strip, open(os.environ["OUT"]+"/log.txt")))]'
```

```bash
tensorboard --logdir "$OUT/summary" --host 127.0.0.1 --port 6006
```

### 中断后恢复

只在运行已停止时执行；用保存的配置和 `last.pth`（`-r`），不与 `-t` 同时使用：

```bash
GPU_IDS=0,1 bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

`last.pth` 只更新到无增强阶段之前（epoch 31），不是最终轮权重。

### 训练结束后：按 val 选权重，评估 test

```bash
python - <<'PY'
import json, os
rows = [json.loads(l) for l in open(os.environ['OUT'] + '/log.txt') if l.strip()]
best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
print('best val AP %.2f at epoch %d (index %d) -> %s' % (100 * best['test_coco_eval_bbox'][0], best['epoch'] + 1,
      best['epoch'], 'best_stg2.pth' if best['epoch'] >= 32 else 'best_stg1.pth'))
PY
```

把上面打印的文件设为 CKPT，然后评估官方 test（11,613 张）：

```bash
export CKPT="$OUT/best_stg2.pth"
export EVAL=$(mktemp -d "$OUT/test_eval_selected_XXXXXX")
ulimit -Sn 65535
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py -c "$OUT/resolved_config.yml" -r "$CKPT" \
  --test-only --seed 0 --use-amp --output-dir "$EVAL" --summary-dir "$EVAL/summary" \
  -u val_dataloader.dataset.img_folder=/data2/zxy/sar/datasets/SARDet_100K/JPEGImages/test \
     val_dataloader.dataset.ann_file=/data2/zxy/sar/datasets/SARDet_100K/Annotations/test.json \
  2>&1 | tee "$EVAL/console.log"
```

### 与 E000-old 对比（总体与按类）

```bash
export BASE=/data2/zxy/sar/experiments/deim_m_sardet100k_deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop_seed0_20261107_083221_byTPNu
ls -d "$BASE"/test_eval_*/
ls "$BASE"/test_eval_*/eval.pth
```

把上面列出的、含 `eval.pth` 的 E000-old test 目录名填入 `BASE_TEST`：

```bash
export BASE_TEST=<E000-old 的 test 评估目录名>
python tools/sardet100k/compare_runs.py \
  --test-ann /data2/zxy/sar/datasets/SARDet_100K/Annotations/test.json \
  --run E000-old="$BASE:$BASE_TEST" --run E020="$OUT:$(basename "$EVAL")" \
  --output "$EVAL/comparison.json"
```

如果 E000-old 没有保存 `eval.pth`，用同一条 test 命令为它补评一次（`CKPT="$BASE/best_stg2.pth"`，`EVAL=$(mktemp -d "$BASE/test_eval_selected_XXXXXX")`，
`-c "$BASE/resolved_config.yml"`），再运行上面的比较。把比较工具的完整输出发回即可。

## 结果（2026-11-19）

用户在旧服务器 GPU 0、1 上运行，36 轮完成，训练耗时 16:02:33。运行目录
`/data2/zxy/sar/experiments/deim_m_sardet100k_deim_hgnetv2_m_sardet100k_vflip_aug_36e_b32_old_server_seed0_20261119_095129_mxMkKY`。
最高 val AP 在第 36 轮（index 35），选 `best_stg2.pth`，EMA 权重。test 为官方 11,613 张图，test 未参与选择；
评估目录 `test_eval_selected_xbw6Iq/`（含 `comparison.json`）。E000-old 的 test 评估目录为 `test_eval_best_stg2_R7nI3O/`，
最高 val 同样在第 36 轮，按类 val AP 取自其 `eval/latest.pth`。数字由 `tools/sardet100k/compare_runs.py` 读出。

| 划分 | 方法 | AP | AP50 | AP75 | APs | APm | APl | AR100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| val | E000-old | 65.15 | 93.22 | 69.97 | 59.88 | 76.70 | 71.64 | 73.14 |
| val | **E020（垂直翻转增强）** | 62.00 | 91.76 | 67.06 | 54.08 | 75.21 | 71.85 | 70.84 |
| val | 差值 | −3.14 | −1.46 | −2.91 | −5.80 | −1.49 | +0.21 | −2.30 |
| test | E000-old | 62.70 | 91.60 | 66.15 | 58.92 | 74.97 | 72.41 | 71.24 |
| test | **E020（垂直翻转增强）** | 58.95 | 88.76 | 62.92 | 53.25 | 72.91 | 71.51 | 68.21 |
| test | 差值 | −3.75 | −2.85 | −3.23 | −5.68 | −2.06 | −0.90 | −3.03 |

按类 AP 相对同服务器基线（E020 − E000-old，与 E013-v2 − E000 并列）：

| 类别 | val E020 | val E013-v2 | test E020 | test E013-v2 |
|---|---:|---:|---:|---:|
| ship | −4.66 | +0.01 | −4.17 | −0.16 |
| aircraft | +0.29 | −0.21 | −1.11 | +1.58 |
| car | −1.57 | −0.01 | −1.45 | +0.12 |
| tank | −8.29 | +0.57 | −10.91 | +1.14 |
| bridge | −5.50 | +7.31 | −4.69 | +6.93 |
| harbor | +0.87 | +4.15 | −0.17 | +4.49 |

逐轮 val AP：第 1 轮 34.60，第 5 轮 45.18，第 10 轮 46.89，第 20 轮 51.91，第 30 轮 59.83，第 32–36 轮 60.70、61.08、61.47、61.75、62.00。

判读（按事先定好的标准，单次运行）：

- **r_val = (62.00 − 65.15) / 1.97 ≈ −1.60，远低于 0.3。** 只做垂直翻转增强不但拿不到 OCEF 的提升，反而在 val 和 test 上分别下降 3.14 和 3.75。
  同样使用垂直翻转，作为训练增强是 −3.75（test），作为网络内保留原方向参照的第二视图（E013-v2）是 +2.35，相差约 6 AP。
- **与 D18 的预测一致：方向增强破坏 SAR 目标稳定的上下方向线索。** 损失集中在小目标（APs −5.80 / −5.68，APl 基本不变）
  和方向性强的类别：tank −8.29 / −10.91、bridge −5.50 / −4.69、ship −4.66 / −4.17。tank 在 D17 中方向效应最大。
- **bridge 的方向正好相反：** 增强使 bridge 下降约 5 AP，OCEF 使它上升约 7 AP。harbor 增强基本持平，OCEF +4 以上。
- 论文用途：消融表中的"垂直翻转增强"一行，按相对同服务器基线的差值与 OCEF 并列；支持"保留原方向作参照、在网络内融合垂直视图，而不是把上下方向当作可交换的增强"。
- 局限：E020 与 E013-v2 不在同一台服务器，只比较各自相对同服务器基线的差值；两台服务器同一基线相差约 0.1–0.4 AP，远小于这里约 6 AP 的差距。
