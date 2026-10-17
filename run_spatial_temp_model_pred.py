import argparse
import heapq
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from monai.data import decollate_batch
from monai.losses.dice import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose
from torch import nn, optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from satimg_dataset_processor.data_generator_pred_torch import (
    TESSERA_CHANNELS,
    FireDataset,
    compute_tessera_stats,
    find_fire_sources,
    fire_name,
    unzip_data_root,
)
from spatial_models.attentionunet import AttentionUnet
from spatial_models.swinunetr.swinunetr import SwinUNETR
from spatial_models.unet import UNet
from spatial_models.unetr.unetr import UNETR


SEED = 42
# TS-SatFire paper, prediction task: "a 1:1 mixture of Dice and CE loss with a
# learning rate of 0.001", CE weight parameter [1.0, 446.7836].
PAPER_CE_WEIGHT = (1.0, 446.7836)
# 27 stored channels become 43 after the 17-class land-cover one-hot.
BASE_MODEL_CHANNELS = 43


# ----- NEW: optional learning-rate schedule ----- #
# NOT PART OF THE TS-SATFIRE PAPER OR THE OFFICIAL CODE: both train with a
# constant learning rate, which stays the default (--lr-schedule none).
LR_SCHEDULES = ("none", "plateau", "cosine")


def build_lr_scheduler(optimizer, schedule, max_epochs, decay_factor, patience, min_lr):
    """NEW, not in the paper: optional once-per-epoch learning-rate decay.

    none:    constant learning rate, as in the paper and the official code.
    plateau: multiply the rate by `decay_factor` after `patience` epochs
             without a lower validation loss, never below `min_lr`.
    cosine:  decay smoothly from the starting rate to `min_lr` over
             `max_epochs` epochs.
    """
    if schedule == "none":
        return None
    if schedule == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=decay_factor,
            patience=patience,
            min_lr=min_lr,
        )
    if schedule == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=min_lr,
        )
    raise ValueError(f"Unsupported learning-rate schedule: {schedule}")


def build_wandb_metric_labels(loss_name):
    """Describe the task, loss, and classes included in each graph."""
    loss_title = {
        "dice": "Dice Loss",
        "dicece": "Weighted Dice + Cross-Entropy Loss",
    }[loss_name]
    labels = {
        "epoch": "Training Epoch (Starting at 0)",
        "train_loss": f"Training {loss_title} — Next-Day Fire Growth",
        "val_loss": f"Validation {loss_title} — Next-Day Fire Growth",
        "miou": "Validation Mean IoU — Background & Fire Growth",
        "mdice": "Validation Mean Dice (F1) — Background & Fire Growth",
        "train_fire_f1": "Training F1 — Next-Day Fire Growth",
        "train_background_f1": "Training F1 — Background",
        "val_fire_f1": "Validation F1 — Next-Day Fire Growth",
        "val_background_f1": "Validation F1 — Background",
        # The paper's IoU (Equation 2) for the fire class; logged, never trained on.
        "train_fire_iou": "Training IoU — Next-Day Fire Growth",
        "val_fire_iou": "Validation IoU — Next-Day Fire Growth",
        # NEW, not in the paper: test fires scored with the current weights
        # every --test-every epochs (logging only).
        "periodic_test_f1": "Test F1 — Next-Day Fire Growth (periodic, current weights)",
        "periodic_test_iou": "Test IoU — Next-Day Fire Growth (periodic, current weights)",
        # Constant unless the NEW (not in the paper) --lr-schedule is used.
        "learning_rate": "Learning Rate",
    }
    if loss_name == "dicece":
        for prefix, split in (("train", "Training"), ("val", "Validation")):
            labels[f"{prefix}_dice_loss"] = (
                f"{split} Weighted Dice Loss — Next-Day Fire Growth"
            )
            labels[f"{prefix}_ce_loss"] = (
                f"{split} Weighted Cross-Entropy Loss — Next-Day Fire Growth"
            )
    return labels


def configure_wandb(
    model_name,
    mode,
    num_heads,
    hidden_size,
    batch_size,
    learning_rate,
    weight_decay,
    max_epochs,
    wandb_user_name,
    run_test,
    use_tessera,
    model_input_channels,
    loss_config,
):
    """Initialize one W&B run for either training or testing."""
    wandb.login()
    wandb.init(project="AFBAPred", entity=wandb_user_name)
    if not run_test:
        metric_labels = build_wandb_metric_labels(loss_config["loss"])
        epoch_metric = metric_labels["epoch"]
        wandb.define_metric(epoch_metric)
        for key, label in metric_labels.items():
            if key != "epoch":
                wandb.define_metric(label, step_metric=epoch_metric)
    wandb.run.name = (
        f"{'test' if run_test else 'train'}_{mode}_{model_name}_"
        f"num_heads_{num_heads}_hidden_size_{hidden_size}_"
        f"batchsize_{batch_size}{'_tessera' if use_tessera else ''}"
        f"{'' if loss_config['loss'] == 'dice' else '_' + loss_config['loss']}"
    )
    wandb.config.update(
        {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": max_epochs,
            "batch_size": batch_size,
            "mode": mode,
            "model": model_name,
            "tessera": use_tessera,
            "input_channels": model_input_channels,
            **loss_config,
        }
    )


def make_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
    """Create a conservative loader that can be tuned after RAM is measured."""
    loader_options = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    # Worker prefetching can hold multiple decompressed windows in RAM.
    # Keep one prefetched batch per worker rather than PyTorch's default two.
    if num_workers > 0:
        loader_options["prefetch_factor"] = 1
        loader_options["persistent_workers"] = True

    return DataLoader(**loader_options)


def validate_dataset_output(dataset, expected_model_channels, dataset_name):
    """Fail before model construction if the Dataset/model shapes disagree."""
    if len(dataset) == 0:
        raise RuntimeError(f"{dataset_name} dataset contains no windows")

    sample = dataset[0]
    data_shape = tuple(sample["data"].shape)
    label_shape = tuple(sample["labels"].shape)
    static_channels = sample["static"].shape[0] if "static" in sample else 0

    if len(data_shape) != 4:
        raise ValueError(
            f"{dataset_name} sample must be (C,T,H,W), found {data_shape}"
        )

    actual_channels = data_shape[0] + static_channels
    if actual_channels != expected_model_channels:
        raise ValueError(
            f"{dataset_name} FireDataset produces {actual_channels} channels "
            f"after preprocessing, but the model is configured for "
            f"{expected_model_channels}. The 27-channel archive plus 17-class "
            f"land-cover expansion gives {BASE_MODEL_CHANNELS} channels; "
            f"--tessera adds {TESSERA_CHANNELS} more."
        )

    print(
        f"{dataset_name} windows={len(dataset)}, "
        f"data shape={data_shape}, label shape={label_shape}"
        + (f", static TESSERA channels={static_channels}" if static_channels else "")
    )


def build_model(
    model_name,
    model_input_channels,
    num_classes,
    ts_length,
    num_heads,
    hidden_size,
    unetr_version,
    swin_temporal_downsample=False,
):
    """Build the requested spatial-temporal model.

    swin_temporal_downsample=False follows the TS-SatFire paper ("we do not
    use downsampling in the temporal dimension in any of our models"); True
    is the released code, whose Swin patch merging also halves an even
    number of days (6 -> 3).
    """
    image_size = (ts_length, 256, 256)

    if model_name == "unet3d":
        return UNet(
            spatial_dims=3,
            in_channels=model_input_channels,
            out_channels=num_classes,
            channels=(64, 128, 256, 512, 1024),
            strides=(1, 2, 2),
        )

    if model_name == "attunet":
        return AttentionUnet(
            spatial_dims=3,
            in_channels=model_input_channels,
            out_channels=num_classes,
            channels=(64, 128, 256, 512, 1024),
            strides=(1, 2, 2),
        )

    if model_name == "unetr3d":
        patch_size = (1, 16, 16)
        kernel_size_up_down = (1, 2, 2)

        if unetr_version == "v0":
            unetr_hidden_size = 768
            mlp_dim = 3072
        else:
            unetr_hidden_size = 384
            mlp_dim = 1536

        return UNETR(
            in_channels=model_input_channels,
            out_channels=num_classes,
            img_size=image_size,
            spatial_dims=3,
            norm_name="batch",
            feature_size=16,
            patch_size=patch_size,
            kernel_size_up_down=kernel_size_up_down,
            hidden_size=unetr_hidden_size,
            mlp_dim=mlp_dim,
        )

    if model_name == "swinunetr3d":
        return SwinUNETR(
            image_size=image_size,
            patch_size=(1, 2, 2),
            window_size=(ts_length, 4, 4),
            in_channels=model_input_channels,
            out_channels=num_classes,
            depths=(2, 2, 2, 2),
            num_heads=(num_heads, num_heads, num_heads, num_heads),
            feature_size=hidden_size,
            norm_name="batch",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            attn_version="v1",
            normalize=True,
            use_checkpoint=False,
            spatial_dims=3,
            temporal_downsample=swin_temporal_downsample,
        )

    raise NotImplementedError(f"Unsupported model: {model_name}")


def batch_inputs(batch, device):
    """Model input on `device`: the 43 stored channels, then (TESSERA mode)
    the static (129, H, W) map repeated across the T days.

    The repeat happens here on the GPU, not in the loader, so host RAM holds
    the TESSERA map once per sample instead of once per day. The channel
    order matches appending TESSERA after the base channels.
    """
    data = batch["data"].to(device, non_blocking=True)
    if "static" in batch:
        static = batch["static"].to(device, non_blocking=True)
        static = static.unsqueeze(2).expand(-1, -1, data.shape[2], -1, -1)
        data = torch.cat((data, static), dim=1)
    return data


def forward_model(model, model_name, data_batch, device):
    """Run the model and reduce its temporal output to one target day."""
    if model_name == "utae":
        data_batch = data_batch.transpose(1, 2).contiguous()
        batch_positions = torch.zeros(data_batch.shape[:2], device=device)
        return model(data_batch, batch_positions=batch_positions)

    outputs = model(data_batch)
    return outputs.mean(dim=2)


def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    save_path,
    extra=None,
    artifact_name=None,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
            # Run config (e.g. TESSERA mode and its normalization) needed to
            # rebuild identical inputs at test time.
            **(extra or {}),
        },
        save_path,
    )

    if artifact_name is not None:
        upload_checkpoint(save_path, epoch, val_loss, artifact_name, extra)


def upload_checkpoint(save_path, epoch, val_loss, artifact_name, extra=None):
    """Store a versioned checkpoint as a W&B model Artifact.

    Online, wait for the upload so a later best checkpoint cannot replace the
    file first; offline runs log it locally for a later `wandb sync`.
    """
    if wandb.run is None:
        raise RuntimeError("A W&B run is required to upload model checkpoints")
    metadata = {"epoch": epoch + 1, "validation_loss": float(val_loss)}
    for key in ("use_tessera", "model_input_channels"):
        if extra and key in extra:
            metadata[key] = extra[key]
    artifact = wandb.Artifact(name=artifact_name, type="model", metadata=metadata)
    artifact.add_file(str(save_path), name=Path(save_path).name)
    uploaded = wandb.run.log_artifact(
        artifact, aliases=["best", "latest", f"epoch-{epoch + 1}"]
    )
    if not wandb.run.offline:
        uploaded.wait()
    print(f"Checkpoint logged to W&B Artifacts: {artifact_name}")


def build_criterion(loss_name, ce_weight=PAPER_CE_WEIGHT, lambda_dice=1.0, lambda_ce=1.0):
    """Training loss for 2-channel (background, fire) logits.

    dice:   sigmoid Dice loss, as in the released TS-SatFire code.
    dicece: the paper's prediction loss, MONAI DiceCELoss with `lambda_dice`
            Dice + `lambda_ce` cross entropy (1:1 by default). MONAI applies
            the `weight` parameter to both terms: per-class Dice weights and
            CrossEntropyLoss class weights (the one-hot target is used as class
            probabilities), the same in MONAI 1.3.2 and later.
    """
    if loss_name == "dice":
        return DiceLoss(include_background=True, reduction="mean", sigmoid=True)
    if loss_name == "dicece":
        return DiceCELoss(
            include_background=True,
            sigmoid=True,
            reduction="mean",
            weight=torch.tensor(ce_weight, dtype=torch.float32),
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )
    raise ValueError(f"Unsupported loss: {loss_name}")


def compute_loss(criterion, outputs, labels):
    """Return (loss, {component name: value}) so components can be logged."""
    if isinstance(criterion, DiceCELoss):
        dice = criterion.dice(outputs, labels)
        ce = criterion.ce(outputs, labels)
        loss = criterion.lambda_dice * dice + criterion.lambda_ce * ce
        return loss, {"dice_loss": dice.detach(), "ce_loss": ce.detach()}
    return criterion(outputs, labels), {}


def _channel_f1(predictions, labels, channel):
    """Return per-sample F1 for one channel of binary (B,2,H,W) masks.

    Empty prediction and empty target score 1, matching existing test F1
    and DiceMetric(ignore_empty=False). Channels are evaluated independently.
    """
    predicted = predictions[:, channel].bool().flatten(start_dim=1)
    target = (labels[:, channel] > 0).flatten(start_dim=1)
    true_positive = (predicted & target).sum(dim=1).float()
    denominator = predicted.sum(dim=1).float() + target.sum(dim=1).float()
    return torch.where(
        denominator > 0,
        2.0 * true_positive / denominator.clamp_min(1.0),
        torch.ones_like(denominator),
    )


def fire_only_f1(predictions, labels):
    """Per-sample next-day fire-growth F1 from thresholded channel 1."""
    return _channel_f1(predictions, labels, channel=1)


def background_f1(predictions, labels):
    """Per-sample background F1 from thresholded channel 0."""
    return _channel_f1(predictions, labels, channel=0)


def fire_only_iou(predictions, labels):
    """Per-sample next-day fire-growth IoU, the paper's Equation 2.

    IoU = TP / (TP + FP + FN) on thresholded channel 1. Empty prediction and
    empty target score 1, like the official code's test, which used
    sklearn's jaccard_score(zero_division=1).
    Logged only; never part of the training loss.
    """
    predicted = predictions[:, 1].bool().flatten(start_dim=1)
    target = (labels[:, 1] > 0).flatten(start_dim=1)
    intersection = (predicted & target).sum(dim=1).float()
    union = (predicted | target).sum(dim=1).float()
    return torch.where(
        union > 0,
        intersection / union.clamp_min(1.0),
        torch.ones_like(union),
    )


def train_model(
    model,
    model_name,
    train_dataset,
    train_dataloader,
    val_dataset,
    val_dataloader,
    optimizer,
    criterion,
    scaler,
    mean_iou,
    dice_metric,
    post_trans,
    device,
    max_epochs,
    top_n_checkpoints,
    checkpoint_name,
    checkpoint_extra=None,
    artifact_name=None,
    lr_scheduler=None,
    test_every=0,
    score_test_fires=None,
):
    """Train and validate; return the checkpoint with the lowest validation loss.

    `lr_scheduler` is the NEW optional learning-rate decay (not in the
    paper); None keeps the paper's constant rate.

    NEW, not in the paper: with `test_every` > 0, `score_test_fires()` scores
    the test fires with the current weights every `test_every` epochs, for
    logging only. It runs in eval mode without gradients, so no weight or
    batch-norm statistic changes, and it never affects checkpoint choice.
    """
    metric_labels = build_wandb_metric_labels(
        "dicece" if isinstance(criterion, DiceCELoss) else "dice"
    )
    # Store (-loss, path), making heap[0] the worst retained checkpoint.
    best_checkpoints = []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        train_components = {}
        train_fire_f1_sum = 0.0
        train_background_f1_sum = 0.0
        train_fire_iou_sum = 0.0
        train_bar = tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc=f"Train epoch {epoch + 1}/{max_epochs}",
        )

        for batch_idx, batch in enumerate(train_bar):
            data_batch = batch_inputs(batch, device)
            labels_batch = batch["labels"].float().to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)
            outputs = forward_model(
                model,
                model_name,
                data_batch,
                device,
            )
            loss, components = compute_loss(criterion, outputs, labels_batch)

            with torch.no_grad():
                predictions = torch.stack(
                    [post_trans(item) for item in decollate_batch(outputs.detach())]
                )
                train_fire_f1_sum += fire_only_f1(predictions, labels_batch).sum().item()
                train_background_f1_sum += background_f1(predictions, labels_batch).sum().item()
                train_fire_iou_sum += fire_only_iou(predictions, labels_batch).sum().item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.detach().item() * data_batch.size(0)
            for name, value in components.items():
                train_components[name] = (
                    train_components.get(name, 0.0)
                    + value.item() * data_batch.size(0)
                )
            running_loss = train_loss / (
                (batch_idx + 1) * data_batch.size(0)
            )
            train_bar.set_postfix(loss=f"{running_loss:.4f}")

            if not np.isfinite(train_loss):
                raise RuntimeError(
                    f"Training loss became non-finite at step {batch_idx}"
                )

        train_loss /= len(train_dataset)
        train_components = {
            f"train_{name}": total / len(train_dataset)
            for name, total in train_components.items()
        }
        wandb.log(
            {
                metric_labels["epoch"]: epoch,
                metric_labels["train_loss"]: train_loss,
                metric_labels["train_fire_f1"]: train_fire_f1_sum / len(train_dataset),
                metric_labels["train_background_f1"]: train_background_f1_sum / len(train_dataset),
                metric_labels["train_fire_iou"]: train_fire_iou_sum / len(train_dataset),
                **{metric_labels[name]: value for name, value in train_components.items()},
            }
        )
        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}"
            + "".join(f", {k}: {v:.4f}" for k, v in train_components.items())
        )

        model.eval()
        val_loss = 0.0
        val_components = {}
        iou_values = []
        dice_values = []
        val_fire_f1_sum = 0.0
        val_background_f1_sum = 0.0
        val_fire_iou_sum = 0.0
        val_bar = tqdm(
            val_dataloader,
            total=len(val_dataloader),
            desc=f"Validation epoch {epoch + 1}/{max_epochs}",
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_bar):
                data_batch = batch_inputs(batch, device)
                labels_batch = batch["labels"].float().to(
                    device,
                    non_blocking=True,
                )

                outputs = forward_model(
                    model,
                    model_name,
                    data_batch,
                    device,
                )
                loss, components = compute_loss(criterion, outputs, labels_batch)
                val_loss += loss.detach().item() * data_batch.size(0)
                for name, value in components.items():
                    val_components[name] = (
                        val_components.get(name, 0.0)
                        + value.item() * data_batch.size(0)
                    )

                discrete_outputs = [
                    post_trans(item)
                    for item in decollate_batch(outputs)
                ]
                discrete_labels = decollate_batch(labels_batch)
                predictions = torch.stack(discrete_outputs)
                val_fire_f1_sum += fire_only_f1(predictions, labels_batch).sum().item()
                val_background_f1_sum += background_f1(predictions, labels_batch).sum().item()
                val_fire_iou_sum += fire_only_iou(predictions, labels_batch).sum().item()

                iou_values.append(
                    mean_iou(
                        discrete_outputs,
                        discrete_labels,
                    ).mean().item()
                )
                dice_values.append(
                    dice_metric(
                        y_pred=discrete_outputs,
                        y=discrete_labels,
                    ).mean().item()
                )

                running_val_loss = val_loss / (
                    (batch_idx + 1) * data_batch.size(0)
                )
                val_bar.set_postfix(loss=f"{running_val_loss:.4f}")

        val_loss /= len(val_dataset)
        val_components = {
            f"val_{name}": total / len(val_dataset)
            for name, total in val_components.items()
        }
        mean_iou_val = float(np.mean(iou_values))
        mean_dice_val = float(np.mean(dice_values))
        val_fire_f1 = val_fire_f1_sum / len(val_dataset)
        val_background_f1 = val_background_f1_sum / len(val_dataset)
        val_fire_iou = val_fire_iou_sum / len(val_dataset)

        wandb.log(
            {
                metric_labels["epoch"]: epoch,
                metric_labels["val_loss"]: val_loss,
                metric_labels["miou"]: mean_iou_val,
                metric_labels["mdice"]: mean_dice_val,
                metric_labels["val_fire_f1"]: val_fire_f1,
                metric_labels["val_background_f1"]: val_background_f1,
                metric_labels["val_fire_iou"]: val_fire_iou,
                metric_labels["learning_rate"]: optimizer.param_groups[0]["lr"],
                **{metric_labels[name]: value for name, value in val_components.items()},
            }
        )
        print(
            f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, "
            + "".join(f"{k}: {v:.4f}, " for k, v in val_components.items())
            + f"Mean IoU: {mean_iou_val:.4f}, "
            f"Mean Dice: {mean_dice_val:.4f}, "
            f"Fire F1: {val_fire_f1:.4f}, Fire IoU: {val_fire_iou:.4f}, "
            f"Background F1: {val_background_f1:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        checkpoint_path = os.path.join(
            wandb.run.dir,
            checkpoint_name.format(epoch=epoch + 1),
        )

        if len(best_checkpoints) < top_n_checkpoints:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                checkpoint_path,
                checkpoint_extra,
                artifact_name,
            )
            heapq.heappush(
                best_checkpoints,
                (-val_loss, checkpoint_path),
            )
        else:
            worst_loss = -best_checkpoints[0][0]
            if val_loss < worst_loss:
                _, removed_path = heapq.heapreplace(
                    best_checkpoints,
                    (-val_loss, checkpoint_path),
                )
                if os.path.exists(removed_path):
                    os.remove(removed_path)
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_loss,
                    checkpoint_path,
                    checkpoint_extra,
                    artifact_name,
                )

        # NEW, not in the paper: step the optional learning-rate schedule.
        if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
        elif lr_scheduler is not None:
            lr_scheduler.step()

        # NEW, not in the paper: watch the test scores during training.
        # Logging only; the test fires never reach the loss or the optimizer.
        if test_every and score_test_fires is not None and (epoch + 1) % test_every == 0:
            # Building test loaders draws random numbers; restoring every
            # random state keeps the training run identical with or without
            # these checks.
            rng_states = (
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            )
            test_scores = score_test_fires()
            random.setstate(rng_states[0])
            np.random.set_state(rng_states[1])
            torch.set_rng_state(rng_states[2])
            if rng_states[3] is not None:
                torch.cuda.set_rng_state_all(rng_states[3])
            wandb.log(
                {
                    metric_labels["epoch"]: epoch,
                    metric_labels["periodic_test_f1"]: test_scores["f1"],
                    metric_labels["periodic_test_iou"]: test_scores["iou"],
                }
            )
            print(
                f"Epoch {epoch + 1}, test fires (current weights, logging only): "
                f"F1 {test_scores['f1']:.4f}, IoU {test_scores['iou']:.4f}"
            )

    print("Best checkpoints:")
    for negative_loss, checkpoint_path in sorted(
        best_checkpoints,
        reverse=True,
    ):
        print(f"loss={-negative_loss:.6f}: {checkpoint_path}")
    # Heap entries are (-loss, path), so the largest is the lowest loss.
    return max(best_checkpoints)[1]


def normalize_for_plot(array):
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum == minimum:
        return np.zeros_like(array, dtype=np.float32)
    return (array - minimum) / (maximum - minimum)


def load_test_ids(test_csv_path, available_ids):
    """Use CSV order when available; otherwise test every available archive."""
    test_csv_path = Path(test_csv_path).expanduser()
    if not test_csv_path.exists():
        print(
            f"Test CSV not found at {test_csv_path}; using NPZ filenames"
        )
        return sorted(available_ids)

    dataframe = pd.read_csv(test_csv_path, dtype={"Id": str})
    ids = dataframe["Id"].astype(str)
    ids = ids[ids != "US_2021_NV3700011641620210517"]
    return [fire_id for fire_id in ids if fire_id in available_ids]


def evaluate_model(
    model,
    model_name,
    test_dataset,
    batch_size,
    num_workers,
    pin_memory,
    post_trans,
    device,
    test_csv_path,
    plot_dir,
    num_heads,
    hidden_size,
    model_input_channels,
    save_plots=True,
    verbose=True,
):
    """Score the test fires one at a time and return the paper's metrics.

    Each fire's score is the mean over its windows; the model's score is the
    mean over fires. Returns {"f1", "iou", "background_f1", "fires"}.
    """
    indices_by_fire = defaultdict(list)
    for dataset_idx, (archive_path, _) in enumerate(test_dataset.index):
        fire_id = fire_name(archive_path)
        if fire_id.startswith("p"):
            fire_id = fire_id[1:]
        indices_by_fire[fire_id].append(dataset_idx)

    test_ids = load_test_ids(test_csv_path, set(indices_by_fire))
    if not test_ids:
        raise RuntimeError("No test fires match the available NPZ archives")

    plot_dir = Path(plot_dir)
    if save_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    f1_all = 0.0
    background_f1_all = 0.0
    iou_all = 0.0
    evaluated_fire_count = 0

    for fire_number, fire_id in enumerate(test_ids):
        fire_subset = Subset(test_dataset, indices_by_fire[fire_id])
        test_dataloader = make_dataloader(
            fire_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        fire_f1 = 0.0
        fire_background_f1 = 0.0
        fire_iou = 0.0
        window_count = 0

        for batch_number, batch in enumerate(test_dataloader):
            data_batch = batch["data"]
            labels_batch = batch["labels"]

            with torch.no_grad():
                outputs = forward_model(
                    model,
                    model_name,
                    batch_inputs(batch, device),
                    device,
                )
                discrete_outputs = [
                    post_trans(item)
                    for item in decollate_batch(outputs)
                ]

            predictions = torch.stack(discrete_outputs).cpu()
            fire_f1 += fire_only_f1(predictions, labels_batch).sum().item()
            fire_background_f1 += background_f1(predictions, labels_batch).sum().item()
            fire_iou += fire_only_iou(predictions, labels_batch).sum().item()
            window_count += data_batch.shape[0]
            if not save_plots:
                continue

            outputs_numpy = predictions.numpy()
            for sample_idx in range(data_batch.shape[0]):
                prediction = outputs_numpy[sample_idx, 1]
                label = (
                    labels_batch[sample_idx, 1] > 0
                ).cpu().numpy()

                background = data_batch[
                    sample_idx,
                    3,
                    -1,
                ].cpu().numpy()
                plt.imshow(
                    normalize_for_plot(background),
                    cmap="gray",
                )

                true_positive = np.where(
                    (prediction == 1) & (label == 1),
                    1.0,
                    np.nan,
                )
                false_positive = np.where(
                    (prediction == 1) & (label == 0),
                    1.0,
                    np.nan,
                )
                false_negative = np.where(
                    (prediction == 0) & (label == 1),
                    1.0,
                    np.nan,
                )

                plt.imshow(
                    true_positive,
                    cmap="autumn",
                    interpolation="nearest",
                )
                plt.imshow(
                    false_positive,
                    cmap="summer",
                    interpolation="nearest",
                )
                plt.imshow(
                    false_negative,
                    cmap="brg",
                    interpolation="nearest",
                )
                plt.axis("off")

                plot_name = (
                    f"id_{fire_id}_nhead_{num_heads}_"
                    f"hidden_{hidden_size}_batch_{batch_number}_"
                    f"sample_{sample_idx}_fire_{fire_number}_"
                    f"nc_{model_input_channels}.png"
                )
                plt.savefig(
                    plot_dir / plot_name,
                    bbox_inches="tight",
                )
                plt.close()

        if window_count == 0:
            print(f"Skipping {fire_id}: no windows")
            continue

        mean_fire_f1 = fire_f1 / window_count
        mean_fire_iou = fire_iou / window_count
        f1_all += mean_fire_f1
        background_f1_all += fire_background_f1 / window_count
        iou_all += mean_fire_iou
        evaluated_fire_count += 1

        if verbose:
            print(f"ID {fire_id} IoU: {mean_fire_iou}")
            print(f"ID {fire_id} F1: {mean_fire_f1}")
            print(f"ID {fire_id} Background F1: {fire_background_f1 / window_count}")

    if evaluated_fire_count == 0:
        raise RuntimeError("No test fires were evaluated")

    model_f1 = f1_all / evaluated_fire_count
    model_iou = iou_all / evaluated_fire_count
    model_background_f1 = background_f1_all / evaluated_fire_count
    if verbose:
        print(
            f"Model F1: {model_f1}; background F1: {model_background_f1}; "
            f"model IoU: {model_iou}"
        )
    return {
        "f1": model_f1,
        "iou": model_iou,
        "background_f1": model_background_f1,
        "fires": evaluated_fire_count,
    }


def log_final_test(scores):
    """Log the test scores of the best-validation checkpoint (the paper's metrics)."""
    test_metrics = {
        "test_f1": scores["f1"],
        "test_iou": scores["iou"],
        "Test F1 — Next-Day Fire Growth": scores["f1"],
        "Test IoU — Next-Day Fire Growth": scores["iou"],
        "Test F1 — Background": scores["background_f1"],
        "test_fires": scores["fires"],
    }
    wandb.log(test_metrics)
    # One value per run: the summary shows it in the run overview and table.
    wandb.run.summary.update(test_metrics)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or test spatial-temporal fire prediction models"
    )
    parser.add_argument("-m", required=True, help="Model name")
    parser.add_argument("-mode", required=True, help="Task name, e.g. pred")
    parser.add_argument("-b", type=int, required=True, help="Batch size")
    parser.add_argument("-r", type=int, required=True, help="Run number")
    parser.add_argument("-lr", type=float, required=True, help="Learning rate")
    parser.add_argument("-nh", type=int, required=True, help="Number of heads")
    parser.add_argument("-ed", type=int, required=True, help="Embedding size")
    parser.add_argument(
        "-nc",
        type=int,
        default=None,
        help="Model input channels after preprocessing: 43, or 172 with "
        "--tessera. Derived when omitted; checked when given.",
    )
    parser.add_argument("-ts", type=int, required=True, help="Time-series length")
    parser.add_argument(
        "-it",
        type=int,
        required=True,
        help="Generation interval; retained for experiment/checkpoint naming",
    )
    parser.add_argument(
        "-test",
        dest="run_test",
        action="store_true",
        help="Run test evaluation instead of training",
    )
    parser.add_argument("-seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--data-root",
        default=None,
        help="Directory containing train/, val/, and test/ NPZ folders",
    )
    parser.add_argument(
        "--unzip-dir",
        default="auto",
        help="Where the window archives are unzipped once, so each window is "
        "read without decompressing (same arrays). 'auto': <data-root>-npy "
        "beside the data root; 'off': read the archives. Skipped when the "
        "data root is already unzipped or the disk lacks space.",
    )
    parser.add_argument(
        "--unetr-version",
        choices=("v0", "v1"),
        default="v1",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path for -test; overrides the generated path",
    )
    parser.add_argument("--load-epoch", type=int, default=199)
    parser.add_argument(
        "--test-csv",
        default=str(Path(__file__).resolve().parent / "roi" / "us_fire_2021_out_new.csv"),
    )
    parser.add_argument("--plot-dir", default="evaluation_plot")
    parser.add_argument(
        "--test-every",
        type=int,
        default=0,
        help="NEW, not in the paper: also score the test fires with the current "
        "weights every N epochs (0 = never). Logging only: no gradients, eval "
        "mode, and checkpoint choice still uses validation loss",
    )
    parser.add_argument(
        "--test-after-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After training, score the test fires with the lowest-validation-loss "
        "checkpoint and log test F1/IoU to the same W&B run",
    )
    parser.add_argument(
        "--tessera",
        action="store_true",
        help="Append per-fire TESSERA embeddings (128 bands + validity mask)",
    )
    parser.add_argument(
        "--tessera-dir",
        default=None,
        help="Folder of tessera/p<id>.npz files; default <data-root>/tessera",
    )
    parser.add_argument(
        "--loss",
        choices=("dice", "dicece"),
        default=None,
        help="dice: released TS-SatFire code. dicece: the paper's prediction "
        "loss, 1:1 Dice + CE with --ce-weight (use with -lr 0.001). Default: "
        "the paper's choice per model, dice for attunet and dicece otherwise",
    )
    parser.add_argument(
        "--swin-temporal-downsample",
        action="store_true",
        help="swinunetr3d: also halve the days in patch merging, as the released "
        "code does. Default follows the paper: no temporal downsampling",
    )
    parser.add_argument(
        "--ce-weight",
        type=float,
        nargs=2,
        default=list(PAPER_CE_WEIGHT),
        metavar=("BACKGROUND", "FIRE"),
        help="dicece class weights (default: the paper's 1.0 446.7836)",
    )
    parser.add_argument("--lambda-dice", type=float, default=1.0)
    parser.add_argument("--lambda-ce", type=float, default=1.0)
    # NEW, not in the TS-SatFire paper or the official code, which both keep
    # the learning rate constant.
    parser.add_argument(
        "--lr-schedule",
        choices=LR_SCHEDULES,
        default="none",
        help="NEW, not in the paper: learning-rate decay. none = constant rate "
        "(paper and official code); plateau = multiply by --lr-decay-factor "
        "after --lr-patience epochs without a lower validation loss; cosine = "
        "smooth decay to --lr-min over --epochs",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=0.5,
        help="NEW, plateau only: multiply the learning rate by this",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help="NEW, plateau only: epochs without a lower validation loss before decaying",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-6,
        help="NEW, plateau and cosine: the lowest learning rate",
    )
    parser.add_argument(
        "--upload-checkpoints",
        action="store_true",
        help="Also store each best checkpoint as a versioned W&B model Artifact",
    )
    parser.add_argument(
        "--tessera-min-coverage",
        type=float,
        default=0.5,
        help="TESSERA pixels below this coverage are zeroed and masked",
    )
    parser.add_argument("--wandb-user", default="gt-fri")
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name = args.m
    mode = args.mode
    if args.loss is None:
        # Paper: Dice+CE for U-Net-3D, UNETR-3D and SwinUNETR-3D; the
        # Attention-U-Net-3D "kept the original setup of using Dice loss".
        args.loss = "dice" if model_name == "attunet" else "dicece"
    batch_size = args.b
    num_heads = args.nh
    hidden_size = args.ed
    ts_length = args.ts
    learning_rate = args.lr
    weight_decay = learning_rate / 10
    max_epochs = args.epochs
    top_n_checkpoints = 1
    num_classes = 2

    # The NPZ members contain 27 channels. FireDataset then replaces the
    # land-cover channel with 17 one-hot channels, producing 43 model channels.
    # TESSERA mode appends 128 embedding bands and one validity mask.
    stored_n_channels = 27
    use_tessera = args.tessera
    model_input_channels = BASE_MODEL_CHANNELS + (
        TESSERA_CHANNELS if use_tessera else 0
    )
    if args.nc is not None and args.nc != model_input_channels:
        raise ValueError(
            f"-nc {args.nc} does not match the preprocessing: expected "
            f"{model_input_channels} {'with' if use_tessera else 'without'} "
            "--tessera"
        )

    root_dir = Path("/content/data")
    processed_data_root = (
        Path(args.data_root).expanduser()
        if args.data_root is not None
        else root_dir
    )
    tessera_dir = None
    if use_tessera:
        tessera_dir = (
            Path(args.tessera_dir).expanduser()
            if args.tessera_dir is not None
            else processed_data_root / "tessera"
        )
    # Unzipping copies only the window archives; TESSERA stays where it is.
    if args.unzip_dir != "off":
        resolved_root = processed_data_root.resolve()
        unzip_root = (
            resolved_root.with_name(resolved_root.name + "-npy")
            if args.unzip_dir == "auto"
            else Path(args.unzip_dir).expanduser()
        )
        processed_data_root = unzip_data_root(
            processed_data_root,
            unzip_root,
            splits=("test",) if args.run_test else ("train", "val", "test"),
        )
    train_npz_dir = processed_data_root / "train"
    val_npz_dir = processed_data_root / "val"
    test_npz_dir = processed_data_root / "test"
    tessera_tag = "_tessera" if use_tessera else ""
    loss_config = {"loss": args.loss}
    if args.loss == "dicece":
        loss_config.update(
            ce_weight=list(args.ce_weight),
            lambda_dice=args.lambda_dice,
            lambda_ce=args.lambda_ce,
        )
    # NEW, not in the paper: the optional learning-rate schedule, recorded in
    # W&B and the checkpoint so decayed runs are never mistaken for paper runs.
    lr_config = {"lr_schedule": args.lr_schedule}
    if args.lr_schedule != "none":
        lr_config["lr_min"] = args.lr_min
    if args.lr_schedule == "plateau":
        lr_config.update(
            lr_decay_factor=args.lr_decay_factor,
            lr_patience=args.lr_patience,
        )
    lr_tag = "" if args.lr_schedule == "none" else f"_lr-{args.lr_schedule}"
    # Name tag for checkpoints; the released-code Dice loss keeps the old names.
    run_tag = tessera_tag + ("" if args.loss == "dice" else f"_{args.loss}") + lr_tag

    configure_wandb(
        model_name=model_name,
        mode=mode,
        num_heads=num_heads,
        hidden_size=hidden_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        wandb_user_name=args.wandb_user,
        run_test=args.run_test,
        use_tessera=use_tessera,
        model_input_channels=model_input_channels,
        loss_config=loss_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # No pinned host copies: with TESSERA a batch of 8 is about 0.8 GiB, and
    # pinning keeps a second copy of each in host RAM.
    pin_memory = False

    model = build_model(
        model_name=model_name,
        model_input_channels=model_input_channels,
        num_classes=num_classes,
        ts_length=ts_length,
        num_heads=num_heads,
        hidden_size=hidden_size,
        unetr_version=args.unetr_version,
        swin_temporal_downsample=args.swin_temporal_downsample,
    )
    if model_name == "swinunetr3d":
        wandb.config.update({"swin_temporal_downsample": args.swin_temporal_downsample})
    wandb.config.update(lr_config)
    # NEW, not in the paper: periodic test scoring during training (logging only).
    wandb.config.update({"test_every": args.test_every})
    if lr_tag:
        wandb.run.name += lr_tag
    model = nn.DataParallel(model)
    model.to(device)

    criterion = build_criterion(
        args.loss,
        ce_weight=args.ce_weight,
        lambda_dice=args.lambda_dice,
        lambda_ce=args.lambda_ce,
    ).to(device)
    mean_iou = MeanIoU(
        include_background=True,
        reduction="mean",
        ignore_empty=False,
    )
    dice_metric = DiceMetric(
        include_background=True,
        reduction="mean",
        ignore_empty=False,
    )
    post_trans = Compose(
        [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler(enabled=device.type == "cuda")

    if not args.run_test:
        tessera_stats = None
        if use_tessera:
            # Normalization comes from the training fires only.
            tessera_stats = compute_tessera_stats(
                [
                    tessera_dir / f"{fire_name(path)}.npz"
                    for path in find_fire_sources(train_npz_dir)
                ],
                min_coverage=args.tessera_min_coverage,
            )
        train_dataset = FireDataset(
            npz_dir=train_npz_dir,
            ts_length=ts_length,
            n_channel=stored_n_channels,
            target_is_single_day=True,
            use_augmentations=True,
            tessera_dir=tessera_dir,
            tessera_stats=tessera_stats,
        )
        val_dataset = FireDataset(
            npz_dir=val_npz_dir,
            ts_length=ts_length,
            n_channel=stored_n_channels,
            target_is_single_day=True,
            use_augmentations=False,
            tessera_dir=tessera_dir,
            tessera_stats=tessera_stats,
        )

        # Use validation for the shape check so random augmentation is avoided.
        validate_dataset_output(
            val_dataset,
            model_input_channels,
            "validation",
        )

        train_dataloader = make_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_dataloader = make_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        # NEW, not in the paper: optional test scoring every --test-every
        # epochs with the current weights, for logging only.
        score_test_fires = None
        if args.test_every > 0:
            if find_fire_sources(test_npz_dir):
                periodic_test_dataset = FireDataset(
                    npz_dir=test_npz_dir,
                    ts_length=ts_length,
                    n_channel=stored_n_channels,
                    target_is_single_day=True,
                    use_augmentations=False,
                    tessera_dir=tessera_dir,
                    tessera_stats=tessera_stats,
                )

                def score_test_fires():
                    return evaluate_model(
                        model=model,
                        model_name=model_name,
                        test_dataset=periodic_test_dataset,
                        batch_size=batch_size,
                        num_workers=args.num_workers,
                        pin_memory=pin_memory,
                        post_trans=post_trans,
                        device=device,
                        test_csv_path=args.test_csv,
                        plot_dir=args.plot_dir,
                        num_heads=num_heads,
                        hidden_size=hidden_size,
                        model_input_channels=model_input_channels,
                        save_plots=False,
                        verbose=False,
                    )
            else:
                print(f"No test archives in {test_npz_dir}; --test-every is ignored")

        checkpoint_name = (
            f"model_{model_name}_mode_{mode}_num_heads_{num_heads}_"
            f"hidden_size_{hidden_size}_batchsize_{batch_size}_"
            "checkpoint_epoch_{epoch}_"
            f"nc_{model_input_channels}_ts_{ts_length}{run_tag}.pth"
        )
        best_checkpoint = train_model(
            model=model,
            model_name=model_name,
            train_dataset=train_dataset,
            train_dataloader=train_dataloader,
            val_dataset=val_dataset,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            mean_iou=mean_iou,
            dice_metric=dice_metric,
            post_trans=post_trans,
            device=device,
            max_epochs=max_epochs,
            top_n_checkpoints=top_n_checkpoints,
            checkpoint_name=checkpoint_name,
            checkpoint_extra={
                "use_tessera": use_tessera,
                "model_input_channels": model_input_channels,
                "swin_temporal_downsample": args.swin_temporal_downsample,
                "tessera_stats": tessera_stats,
                "loss_config": loss_config,
                "lr_config": lr_config,
            },
            artifact_name=(
                f"{model_name}{run_tag.replace('_', '-')}-checkpoints-{wandb.run.id}"
                if args.upload_checkpoints
                else None
            ),
            lr_scheduler=build_lr_scheduler(
                optimizer,
                args.lr_schedule,
                max_epochs,
                args.lr_decay_factor,
                args.lr_patience,
                args.lr_min,
            ),
            test_every=args.test_every,
            score_test_fires=score_test_fires,
        )
        if not args.test_after_train:
            return
        if not find_fire_sources(test_npz_dir):
            print(f"No test archives in {test_npz_dir}; skipping the test step")
            return
        # The paper's metrics on the test fires, from the best-validation checkpoint.
        checkpoint_path = best_checkpoint
    else:
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            checkpoint_path = (
                Path("saved_models")
                / (
                    f"model_{model_name}_mode_{mode}_num_heads_{num_heads}_"
                    f"hidden_size_{hidden_size}_batchsize_{batch_size}_"
                    f"checkpoint_epoch_{args.load_epoch}_"
                    f"nc_{model_input_channels}_ts_{ts_length}{run_tag}.pth"
                )
            )

    # weights_only=False: the checkpoint also stores the TESSERA statistics.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    # Checkpoints from before TESSERA support carry no flag: they are 43-channel.
    trained_with_tessera = bool(checkpoint.get("use_tessera", False))
    if trained_with_tessera != use_tessera:
        raise ValueError(
            f"{checkpoint_path} was trained "
            f"{'with' if trained_with_tessera else 'without'} TESSERA; "
            f"{'add' if trained_with_tessera else 'remove'} --tessera"
        )
    # Checkpoints from before this flag used the released code's merging.
    trained_temporal_downsample = bool(checkpoint.get("swin_temporal_downsample", True))
    if model_name == "swinunetr3d" and trained_temporal_downsample != args.swin_temporal_downsample:
        raise ValueError(
            f"{checkpoint_path} was trained "
            f"{'with' if trained_temporal_downsample else 'without'} temporal "
            f"downsampling; {'add' if trained_temporal_downsample else 'remove'} "
            "--swin-temporal-downsample"
        )

    test_dataset = FireDataset(
        npz_dir=test_npz_dir,
        ts_length=ts_length,
        n_channel=stored_n_channels,
        target_is_single_day=True,
        use_augmentations=False,
        tessera_dir=tessera_dir,
        tessera_stats=checkpoint.get("tessera_stats"),
    )
    validate_dataset_output(
        test_dataset,
        model_input_channels,
        "test",
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(
        f"Loaded checkpoint epoch={checkpoint['epoch']} "
        f"loss={checkpoint['loss']}"
    )

    final_scores = evaluate_model(
        model=model,
        model_name=model_name,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        post_trans=post_trans,
        device=device,
        test_csv_path=args.test_csv,
        plot_dir=args.plot_dir,
        num_heads=num_heads,
        hidden_size=hidden_size,
        model_input_channels=model_input_channels,
    )
    log_final_test(final_scores)


if __name__ == "__main__":
    main()
    wandb.finish()
