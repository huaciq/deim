# DEIM research working agreement

This repository is the experimental baseline for an academic paper. Preserve
reproducibility, fair baseline comparisons, existing results, and user changes.
Use new configs for new methods/ablations rather than silently changing the
settings behind an existing reported result.

## Research execution policy

- 本仓库是论文研究基座，后续模型改进、消融实验及训练方案修改均在此基础上持续推进。
- 在用户授权的研究范围内直接完成必要修改和实验交接，不加入防御性操作，不因假设性风险过于保守，也不增加无必要的确认流程。
- 禁止一切哈希校验、哈希比对、哈希审计及冒烟测试（smoke test），不得将这些操作作为修改、交付或启动训练的前置条件。
- 验证仅围绕具体改动所需的功能正确性和论文实验可复现性开展，不额外堆叠防御性检查。记录 Git revision 用于实验溯源，不属于文件哈希校验。
- 每次涉及实验代码或配置的修改完成后，必须给出 Linux 服务器上通过 Git 更新代码、启动训练、恢复训练以及查看日志和指标的可复制命令；不能仅交付本地修改。

## Environments and transfer

- Windows development repository: `F:\DEIM`; use PowerShell and `rg`.
- Local Python default: Conda `pytorch` unless explicitly overridden.
- Server repository: `/home/zxy/sar/repos/deim`.
- Git remote: `origin`, `https://github.com/huaciq/deim.git`; integration branch: `main`.
- Direct SSH from the development environment is unavailable. Transfer source
  changes through Git. Do not assume local edits exist on the server.
- Server Conda environment: `deim_zxy` (`conda activate deim_zxy`).
- Server GPUs: two RTX 4090, 24 GB each. Default to
  `CUDA_VISIBLE_DEVICES=0,1` and `torchrun --standalone --nproc_per_node=2`.
  Use one GPU only when the user explicitly requests it.
- Dataset root convention: `/home/zxy/sar/datasets`. The user-confirmed OGSOD
  exception is `/data2/zxy/sar/datasets/OGSOD-1.0/sar`: train images in
  `images/train`, validation images in `images/test`, annotations in
  `RTDETR_COCO/train.json` and `RTDETR_COCO/val.json`.
- Experiment output root: `/home/zxy/sar/experiments`.
- Formal training runs use `nohup`, background execution, saved launcher PID,
  and `disown`. Redirect stdin from `/dev/null` and stdout/stderr to a log file.

## Experimental implementation and verification

Inspect Git status before changes. Keep edits scoped, preserve unrelated edits,
and verify relevant functionality when needed for the specific change, without
hash checks or smoke tests.
Record config, Git revision, seed, global batch size, input size, learning rate,
epochs, augmentation, EMA, AMP, initialization weights and evaluation protocol.
`total_batch_size` is global across GPUs. Do not multiply it again for two GPUs.
Never commit datasets, weights, experiment logs, caches, or generated results.
Never overwrite another run or modify original annotations without authorization.

The actual entry point is root `train.py`. Use `-t` for COCO initialization and
`-r` for restoring a run; do not supply both. OGSOD baseline uses the checkpoint
`/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth`.
With `remap_mscoco_category: False`, category IDs are training labels and must
be zero-based. The baseline uses original COCO annotations with split prefixes
in `file_name`, so `img_folder` is the `sar` root for both splits. The preparation
script is optional; do not require it when original annotations already work.

## Required handoff after changes

Provide changed files and intent, local validation and its limits, branch/commit
status, exact server Git update commands, environment activation, any data
preparation, and copy-paste training/evaluation and resume commands. Include
exact console log, GPU/process monitoring, metric and TensorBoard commands.
Default formal runs to both GPUs with `nohup ... > console.log 2>&1 < /dev/null &`,
then save `$!` before calling `disown`. Choose a fresh experiment directory.
Do not claim remote execution or successful full training without evidence.

Detection training writes JSONL epoch metrics to `log.txt`, TensorBoard events
to `summary/`, and COCO evaluation data to `eval/latest.pth`. Checkpoints are
`best_stg1.pth` and (when saved) `best_stg2.pth`. Current solver updates `last.pth`
only before the no-augmentation stage; do not describe it as the final epoch.
Select the best checkpoint by evaluated AP, not just the stage filename.
COCO bbox metric indices 0, 1, 2 are AP@[.50:.95], AP50, AP75; values are in
0..1, multiply by 100 for percentage reporting. Recheck these behaviors after
solver changes. On reconnect, explicitly set `OUT` and activate `deim_zxy`.

Use safe Git fetch/switch/pull with `--ff-only`. If the server worktree is dirty
or diverged, inspect it; never recommend destructive reset or forced checkout.
