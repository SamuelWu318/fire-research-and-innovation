import argparse
import heapq
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from monai.data import decollate_batch
from monai.losses.dice import DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import Activations, AsDiscrete, Compose
from sklearn.metrics import f1_score, jaccard_score
from torch import nn, optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from satimg_dataset_processor.data_generator_pred_torch import FireDataset
from spatial_models.attentionunet import AttentionUnet
from spatial_models.swinunetr.swinunetr import SwinUNETR
from spatial_models.unet import UNet
from spatial_models.unetr.unetr import UNETR


SEED = 42


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
):
    """Initialize one W&B run for either training or testing."""
    wandb.login()
    wandb.init(project="AFBAPred", entity=wandb_user_name)
    wandb.run.name = (
        f"{'test' if run_test else 'train'}_{mode}_{model_name}_"
        f"num_heads_{num_heads}_hidden_size_{hidden_size}_"
        f"batchsize_{batch_size}"
    )
    wandb.config.update(
        {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": max_epochs,
            "batch_size": batch_size,
            "mode": mode,
            "model": model_name,
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

    if len(data_shape) != 4:
        raise ValueError(
            f"{dataset_name} sample must be (C,T,H,W), found {data_shape}"
        )

    actual_channels = data_shape[0]
    if actual_channels != expected_model_channels:
        raise ValueError(
            f"{dataset_name} FireDataset produces {actual_channels} channels "
            f"after preprocessing, but the model is configured for "
            f"{expected_model_channels}. The current 27-channel archive plus "
            "17-class land-cover expansion should produce 43 channels."
        )

    print(
        f"{dataset_name} windows={len(dataset)}, "
        f"data shape={data_shape}, label shape={label_shape}"
    )


def build_model(
    model_name,
    model_input_channels,
    num_classes,
    ts_length,
    num_heads,
    hidden_size,
    unetr_version,
):
    """Build the requested spatial-temporal model."""
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
        )

    raise NotImplementedError(f"Unsupported model: {model_name}")


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
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
        },
        save_path,
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
):
    """Train and validate while retaining the best validation checkpoints."""
    # Store (-loss, path), making heap[0] the worst retained checkpoint.
    best_checkpoints = []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc=f"Train epoch {epoch + 1}/{max_epochs}",
        )

        for batch_idx, batch in enumerate(train_bar):
            data_batch = batch["data"].to(device, non_blocking=True)
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
            loss = criterion(outputs, labels_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.detach().item() * data_batch.size(0)
            running_loss = train_loss / (
                (batch_idx + 1) * data_batch.size(0)
            )
            train_bar.set_postfix(loss=f"{running_loss:.4f}")

            if not np.isfinite(train_loss):
                raise RuntimeError(
                    f"Training loss became non-finite at step {batch_idx}"
                )

        train_loss /= len(train_dataset)
        wandb.log({"epoch": epoch, "train_loss": train_loss})
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        iou_values = []
        dice_values = []
        val_bar = tqdm(
            val_dataloader,
            total=len(val_dataloader),
            desc=f"Validation epoch {epoch + 1}/{max_epochs}",
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_bar):
                data_batch = batch["data"].to(
                    device,
                    non_blocking=True,
                )
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
                loss = criterion(outputs, labels_batch)
                val_loss += loss.detach().item() * data_batch.size(0)

                discrete_outputs = [
                    post_trans(item)
                    for item in decollate_batch(outputs)
                ]
                discrete_labels = decollate_batch(labels_batch)

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
        mean_iou_val = float(np.mean(iou_values))
        mean_dice_val = float(np.mean(dice_values))

        wandb.log(
            {
                "val_loss": val_loss,
                "miou": mean_iou_val,
                "mdice": mean_dice_val,
            }
        )
        print(
            f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, "
            f"Mean IoU: {mean_iou_val:.4f}, "
            f"Mean Dice: {mean_dice_val:.4f}"
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
                )

    print("Best checkpoints:")
    for negative_loss, checkpoint_path in sorted(
        best_checkpoints,
        reverse=True,
    ):
        print(f"loss={-negative_loss:.6f}: {checkpoint_path}")


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
):
    """Evaluate one fire at a time while sharing one indexed test Dataset."""
    indices_by_fire = defaultdict(list)
    for dataset_idx, (archive_path, _) in enumerate(test_dataset.index):
        fire_id = archive_path.stem
        if fire_id.startswith("p"):
            fire_id = fire_id[1:]
        indices_by_fire[fire_id].append(dataset_idx)

    test_ids = load_test_ids(test_csv_path, set(indices_by_fire))
    if not test_ids:
        raise RuntimeError("No test fires match the available NPZ archives")

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    f1_all = 0.0
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
        fire_iou = 0.0
        window_count = 0

        for batch_number, batch in enumerate(test_dataloader):
            data_batch = batch["data"]
            labels_batch = batch["labels"]

            with torch.no_grad():
                outputs = forward_model(
                    model,
                    model_name,
                    data_batch.to(device, non_blocking=True),
                    device,
                )
                discrete_outputs = [
                    post_trans(item)
                    for item in decollate_batch(outputs)
                ]

            outputs_numpy = torch.stack(discrete_outputs).cpu().numpy()
            window_count += data_batch.shape[0]

            for sample_idx in range(data_batch.shape[0]):
                prediction = outputs_numpy[sample_idx, 1]
                label = (
                    labels_batch[sample_idx, 1] > 0
                ).cpu().numpy()

                fire_f1 += f1_score(
                    label.flatten(),
                    prediction.flatten(),
                    zero_division=1.0,
                )
                fire_iou += jaccard_score(
                    label.flatten(),
                    prediction.flatten(),
                    zero_division=1.0,
                )

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
        iou_all += mean_fire_iou
        evaluated_fire_count += 1

        print(f"ID {fire_id} IoU: {mean_fire_iou}")
        print(f"ID {fire_id} F1: {mean_fire_f1}")

    if evaluated_fire_count == 0:
        raise RuntimeError("No test fires were evaluated")

    model_f1 = f1_all / evaluated_fire_count
    model_iou = iou_all / evaluated_fire_count
    print(f"Model F1: {model_f1}; model IoU: {model_iou}")
    wandb.log({"test_f1": model_f1, "test_iou": model_iou})


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
        required=True,
        help="Model input channels after preprocessing; currently 43",
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
        default="~/CalFireMonitoring/roi/us_fire_2021_out_new.csv",
    )
    parser.add_argument("--plot-dir", default="evaluation_plot")
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
    stored_n_channels = 27
    model_input_channels = args.nc
    if model_input_channels != 43:
        raise ValueError(
            f"-nc must be 43 with the current FireDataset preprocessing; "
            f"received {model_input_channels}"
        )

    root_dir = Path("/content/data")
    processed_data_root = (
        Path(args.data_root).expanduser()
        if args.data_root is not None
        else root_dir
    )
    train_npz_dir = processed_data_root / "train"
    val_npz_dir = processed_data_root / "val"
    test_npz_dir = processed_data_root / "test"

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
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    model = build_model(
        model_name=model_name,
        model_input_channels=model_input_channels,
        num_classes=num_classes,
        ts_length=ts_length,
        num_heads=num_heads,
        hidden_size=hidden_size,
        unetr_version=args.unetr_version,
    )
    model = nn.DataParallel(model)
    model.to(device)

    criterion = DiceLoss(
        include_background=True,
        reduction="mean",
        sigmoid=True,
    )
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
        train_dataset = FireDataset(
            npz_dir=train_npz_dir,
            ts_length=ts_length,
            n_channel=stored_n_channels,
            target_is_single_day=True,
            use_augmentations=True,
        )
        val_dataset = FireDataset(
            npz_dir=val_npz_dir,
            ts_length=ts_length,
            n_channel=stored_n_channels,
            target_is_single_day=True,
            use_augmentations=False,
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

        checkpoint_name = (
            f"model_{model_name}_mode_{mode}_num_heads_{num_heads}_"
            f"hidden_size_{hidden_size}_batchsize_{batch_size}_"
            "checkpoint_epoch_{epoch}_"
            f"nc_{model_input_channels}_ts_{ts_length}.pth"
        )
        train_model(
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
        )
        return

    test_dataset = FireDataset(
        npz_dir=test_npz_dir,
        ts_length=ts_length,
        n_channel=stored_n_channels,
        target_is_single_day=True,
        use_augmentations=False,
    )
    validate_dataset_output(
        test_dataset,
        model_input_channels,
        "test",
    )

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = (
            Path("saved_models")
            / (
                f"model_{model_name}_mode_{mode}_num_heads_{num_heads}_"
                f"hidden_size_{hidden_size}_batchsize_{batch_size}_"
                f"checkpoint_epoch_{args.load_epoch}_"
                f"nc_{model_input_channels}_ts_{ts_length}.pth"
            )
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(
        f"Loaded checkpoint epoch={checkpoint['epoch']} "
        f"loss={checkpoint['loss']}"
    )

    evaluate_model(
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


if __name__ == "__main__":
    main()
