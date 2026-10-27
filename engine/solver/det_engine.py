"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import math
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def _to_float32(value):
    """Convert AMP outputs for the criterion without detaching their gradients."""
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: _to_float32(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_float32(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_float32(item) for item in value)
    return value


def _outputs_are_finite(outputs, device):
    """Agree across ranks before replaying a forward that used collectives."""
    checks = []

    def collect(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            checks.append(torch.isfinite(value.detach()).all().to(device))
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(outputs)
    finite = torch.stack(checks).all().to(torch.int32)
    if dist_utils.is_dist_available_and_initialized():
        torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
    return bool(finite.item())


def _snapshot_forward_state(model, device):
    # Forward may update BN buffers and consume dropout/denoising random numbers.
    buffers = [(value, value.detach().clone()) for value in model.buffers()]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    return buffers, cpu_rng, cuda_rng


def _training_image_ids(targets):
    """Keep source IDs grouped by sample, including Mosaic's four image IDs."""
    return [target['image_id'].detach().reshape(-1).cpu().tolist()
            for target in targets]


def _module_training_metrics(model):
    """Expose residual scales without adding diagnostic tensors to the loss."""
    module = dist_utils.de_model(model)
    values = {}
    for owner, name in [('encoder', 'dsg'), ('decoder', 'slqc')]:
        branch = getattr(getattr(module, owner, None), name, None)
        if branch is not None:
            values[name + '_alpha'] = branch.alpha.detach().float().item()
    return values


def _retry_forward_fp32(model, samples, targets, state, device, epoch, step, reason):
    buffers, cpu_rng, cuda_rng = state
    with torch.no_grad():
        for value, saved_value in buffers:
            value.copy_(saved_value)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)
    image_ids = _training_image_ids(targets)
    print(f'AMP_FP32_RETRY epoch={epoch} step={step} reason={reason} '
          f'image_ids={image_ids}; same batch, CPU saved activations')
    # Only this exceptional batch uses FP32. Offload saved activations rather
    # than changing batch size or splitting the batch (which would change BN).
    with torch.autocast(device_type=device.type, enabled=False), \
            torch.autograd.graph.save_on_cpu(pin_memory=True):
        return model(samples, targets=targets)


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)
    amp_dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}[
        kwargs.get('amp_dtype', 'float16')]
    amp_loss_fp32 = kwargs.get('amp_loss_fp32', True)
    amp_fp32_retry = kwargs.get('amp_fp32_retry', False)
    if amp_fp32_retry and (scaler is None or not scaler.is_enabled()
                           or amp_dtype != torch.float16):
        raise ValueError('AMP FP32 replay requires FP16 AMP with GradScaler enabled')

    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))
        forward_state = _snapshot_forward_state(model, device) if amp_fp32_retry else None
        retried_fp32 = False

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, cache_enabled=True):
                outputs = model(samples, targets=targets)
            # Disabling autocast alone does not convert FP16/BF16 tensors.
            # This includes auxiliary and denoising outputs used by FGL/DDF.
            if amp_loss_fp32:
                outputs = _to_float32(outputs)
        else:
            outputs = model(samples, targets=targets)

        if amp_fp32_retry and not _outputs_are_finite(outputs, device):
            del outputs  # Release the invalid graph before restoring BN buffers.
            outputs = _retry_forward_fp32(
                model, samples, targets, forward_state, device, epoch, i, 'nonfinite_output')
            retried_fp32 = True
            if not _outputs_are_finite(outputs, device):
                raise FloatingPointError(
                    f'FP32 replay still has non-finite outputs: epoch={epoch}, step={i}')

        if not torch.isfinite(outputs['pred_boxes']).all():
            image_ids = _training_image_ids(targets)
            raise FloatingPointError(
                f'Non-finite pred_boxes before matching: epoch={epoch}, step={i}, '
                f'amp_dtype={amp_dtype}, image_ids={image_ids}')

        with torch.autocast(device_type=device.type, enabled=False):
            loss_dict = criterion(outputs, targets, **metas)
        loss = sum(loss_dict.values())

        # Check the original losses before backward/optimizer/EMA, rather than
        # after updating the training state. All ranks inspect reduced losses.
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        if not math.isfinite(loss_value) and amp_fp32_retry and not retried_fp32:
            # Criterion caches are cleared by its forward. Release the old graph
            # before restoring buffers that it may have saved for backward.
            outputs = loss_dict = loss = loss_dict_reduced = loss_value = None
            outputs = _retry_forward_fp32(
                model, samples, targets, forward_state, device, epoch, i, 'nonfinite_loss')
            retried_fp32 = True
            if not _outputs_are_finite(outputs, device):
                raise FloatingPointError(
                    f'FP32 replay still has non-finite outputs: epoch={epoch}, step={i}')
            with torch.autocast(device_type=device.type, enabled=False):
                loss_dict = criterion(outputs, targets, **metas)
            loss = sum(loss_dict.values())
            loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
            loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            raise FloatingPointError(
                f'Non-finite loss before backward: epoch={epoch}, step={i}, '
                f'losses={loss_dict_reduced}')

        if scaler is not None:
            scaler.scale(loss).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                # FP16 GradScaler handles overflow by skipping the update.
                # BF16 uses a disabled scaler, so reject invalid gradients here.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm,
                    error_if_nonfinite=not scaler.is_enabled())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        branch_metrics = _module_training_metrics(model)
        mpq_loss = getattr(criterion, 'mpq_loss', None)
        if mpq_loss is not None:
            mpq_metrics = dist_utils.reduce_dict(mpq_loss.last_metrics)
            branch_metrics.update({name: value.item() for name, value in mpq_metrics.items()})
        metric_logger.update(**branch_metrics)
        if 'pred_scs' in outputs:
            validity = torch.cat([target['scs_valid'] for target in targets])
            metric_logger.update(scs_valid_fraction=validity.float().mean().item() if validity.numel() else 0.0)
        if amp_fp32_retry:
            metric_logger.update(fp32_retry_rate=float(retried_fp32))
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for name, value in branch_metrics.items():
                writer.add_scalar('Module/' + name, value, global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator
