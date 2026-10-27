# MPQ query discrimination: protocol and launch

## Scope and fixed comparison

MPQ-single is experiment B of the 2026-10-27 review. It tests a semantic auxiliary loss on the last ordinary decoder queries; seven classes each have one prototype. MPQ-other4 is experiment C and changes only the number of centers of category 6 (other) from one to four. Launch C separately after B; no queue or automatic second run is created.

The detector is DEIM-D-FINE-M / HGNetv2-B2, SAR-AIRcraft-1.0 (train3047/val442/test879). Category IDs: 0 A220, 1 A320/321, 2 A330, 3 ARJ21, 4 Boeing737, 5 Boeing787, 6 other. The source dataset and annotations are reused without preparation or modification. Reference baseline: /root/zxy/experiments/deim_m_sar_aircraft_102e_b28_nozoom_nocrop_seed0_20261026_142643_LMw1Yy, best AP64.510414 at logical epoch90 (0-based), git a448005.

Frozen training: seed0, GPU0/1, 102 epochs, global batch28 (14/GPU), COCO initialization via -t /root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth. AdamW lr3.5e-4, backbone non-BN3.5e-5, weight decay1e-4, Norm/BN zero decay, betas(.9,.999). Flatcosine warmup2000 iterations, flat_epoch49, no_aug_epoch12, lr_gamma.5. Base/eval640; training480..800 step32 until epoch90. RandomZoomOut and RandomIoUCrop disabled; outer Mosaic prob.5 and MixUp.5 in [4,49), PhotometricDistort.5 in [4,90), horizontal flip.5. Mosaic operator internal probability remains1.0; the outer scheduler controls its application. EMA.9999/warmup1000, SyncBN, AMP fp16 with FP32 loss and existing exceptional retry, clip norm.1. Validation: FP32, top300, no NMS, full COCO AP, global valbatch64. The held-out test is not used to select the method.

## Method and exact objective

Last ordinary Query q -> LayerNorm -> bias-free Linear(256,128) -> L2 normalization z. Trainable prototypes are L2 normalized. For each class c with Kc centers pck, compute rick=z dot pck, aick=softmax_k(rick/0.1), sic=sum_k(aick*rick), and auxiliary CE(s_i/0.1, GT class). No prototype score is substituted into pred_logits, matching, LQE, FDR, DN or auxiliary decoder supervision. The existing matcher supplies one-to-one final-layer ordinary matches for MPQ. The projection and centers are model parameters, hence optimized, EMA-saved and DDP-synchronized with the detector. Their initialization uses an isolated RNG scope to preserve common baseline initialization and subsequent random sequence.

With detached matched IoU u, include u>=.7 and weight u^2. Within each class, sum(weight*CE)/sum(weight), then average over classes with any eligible instance in the global two-GPU batch. Denominators are globally reduced and detached; local numerators are multiplied by world size because DDP averages gradients. Empty ranks and classes retain a connected zero-loss path. The effective coefficient is:

    0.05 * clamp((epoch + step/steps_per_epoch - 49)/5, 0, 1)

Epochs are the JSONL/solver's logical 0-based epoch, not the checkpoint's internal last_epoch. Loss is exactly zero before epoch49, ramps during [49,54), and reaches .05 at epoch54. The head remains in the graph before activation so find_unused_parameters=False is valid. Zero gradients still let AdamW maintain its normal state and decay the head's non-norm parameters; they do not supply a semantic gradient to the detector. At evaluation the head is not executed, and deploy() removes it.

Sources: FSCE https://arxiv.org/abs/2103.05950 and SoftTriple https://arxiv.org/abs/1909.05235 . This is an adaptation and hypothesis, not a demonstrated performance claim. Auxiliary gradients still affect shared queries/features, so AP75 and localization must be checked.

## Diagnostics and decision criteria

log.txt writes train_loss_mpq (already weighted), train_mpq_ce (unweighted class-balanced CE), train_mpq_ramp, train_mpq_valid_fraction, train_mpq_valid_c0..c6 (global eligible counts per batch, epoch-averaged), and train_mpq_usage_pN (own-class soft center occupancy among eligible matches). For single centers usage is one for present classes and zero for absent classes. All these diagnostics are detached and excluded from total loss. TensorBoard writes Loss/loss_mpq and Module/mpq_*.

Use the same checkpoint selected by best validation AP for AP50/AP75/AR100, per-class AP and input640 size-group AP. Focus on A220/ARJ21/other confusions and the 32..64px group. Compare fixed score>=.3 and equal-recall errors. The original-image COCO small group has only one GT and cannot validate a small-object claim. Do not select a method only from early training or auxiliary loss. If B/C improve, repeat matched baseline/method seeds1 and2; do not combine old branches before standalone results.

## Server source update

Actual user-authorized server checkout is /root/zxy/deim, Conda deim_zxy. Never reset or overwrite unrelated server changes. Inspect status and update main by fast-forward only:

~~~bash
cd /root/zxy/deim
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
~~~

If origin cannot be reached, the delivered Git bundle is an equivalent source transfer. Fetch the exact bundle path recorded in the launch handoff, then git merge --ff-only FETCH_HEAD. Existing untracked OGSOD root configs are unrelated and must be preserved.

## Fresh run (both cards checked once immediately before launch)

~~~bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
bash tools/run_sar_module_experiment.sh configs/deim_dfine/sar_aircraft_mpq/mpq_single.yml
~~~

This records resolved config/revision/protocol/seed/weights and GPU state, creates a fresh output directory, launches torchrun --standalone --nproc_per_node=2 with CUDA_VISIBLE_DEVICES=0,1 using nohup, redirects stdin from /dev/null and output to console.log, saves launcher.pid, and calls disown. It refuses to launch if either card is busy; it neither stops jobs nor queues work. Use the returned actual OUT for all commands below. MPQ-other4 is available as configs/deim_dfine/sar_aircraft_mpq/mpq_other4.yml but is not launched by the single experiment command.

## Monitor, evaluate, and resume

Set OUT explicitly after reconnect. The exact actual directory is in the launch handoff accompanying this file.

~~~bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/REPLACE_WITH_ACTUAL_RUN
tail -n 80 -f "$OUT/console.log"
nvidia-smi -i 0,1
ps -fp "$(cat "$OUT/launcher.pid")"
tail -n 1 "$OUT/log.txt"
python -c 'import json,sys; rows=[json.loads(s) for s in open(sys.argv[1]) if s.strip()]; r=max(rows,key=lambda x:x["test_coco_eval_bbox"][0]); m=r["test_coco_eval_bbox"]; print({"epoch":r["epoch"],"AP":m[0]*100,"AP50":m[1]*100,"AP75":m[2]*100,"AR100":m[8]*100})' "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
~~~

Evaluate a selected checkpoint (choose by evaluated AP, not only the stage name):

~~~bash
cd /root/zxy/deim
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/best_stg2.pth" --test-only
~~~

Resume only after the run has stopped, using the exact saved config and last.pth (which is only refreshed before epoch90 in the existing solver). The launcher checks both GPUs again and saves a separate resume console log:

~~~bash
cd /root/zxy/deim
bash tools/run_sar_module_experiment.sh --resume "$OUT"
~~~

Resume restores model/prototypes, optimizer, EMA and the logical ramp is recomputed from training epoch/step. Do not supply -t together with -r. Early epochs have near-zero AP due to initial COCO-to-seven-class adaptation and are not evidence of a new module failing.

## Targeted functional verification

Local CPU (pytorch environment): python -m unittest discover -s tests -p test_query_prototypes.py -v . Tests verify the actual loss equations, detached IoU, nonzero query/prototype gradients after activation, connected zeros before activation/empty matches, unchanged detector outputs and DN separation, FP32 head math under AMP, and protocol equality. Server CPU: torchrun --standalone --nproc_per_node=2 tests/check_mpq_ddp.py verifies actual two-rank gradients against explicit serial global class means for unequal class counts, an empty rank, and pre-activation. Formal training supplies the actual full-detector AMP/GPU execution evidence.
