import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Validate a TESSERA-augmented dataset.")
    parser.add_argument("path", help="Path to the augmented [N, C, T, H, W] .npy dataset.")
    parser.add_argument(
        "--original-path",
        help="Optional unaugmented [N, 8, T, H, W] dataset to verify original channels are preserved.",
    )
    args = parser.parse_args()

    x = np.load(args.path, mmap_mode="r")
    print(f"Dataset shape: {x.shape}")
    if x.ndim != 5:
        raise ValueError(f"Expected [N, C, T, H, W] 5D data, got {x.shape}.")
    if x.shape[1] != 136:
        raise ValueError(f"Expected 8 original + 128 TESSERA channels (136 total), got {x.shape[1]}.")
    if x.shape[2] < 1 or min(x.shape[3:]) < 1:
        raise ValueError(f"Dataset has an empty time or spatial dimension: {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("Dataset contains NaN or infinite values.")

    # Embeddings are static and are copied into every timestep at every pixel.
    tessera = x[:, 8:, :, :, :]
    if not np.allclose(tessera, tessera[:, :, :1, :, :]):
        raise ValueError("TESSERA values differ across timesteps for at least one sample/channel/pixel.")
    print("TESSERA block is finite and constant across timesteps.")

    if args.original_path:
        original = np.load(args.original_path, mmap_mode="r")
        expected_shape = (x.shape[0], 8, *x.shape[2:])
        if original.shape != expected_shape:
            raise ValueError(f"Original dataset shape {original.shape} does not match expected {expected_shape}.")
        if not np.array_equal(x[:, :8, :, :, :], original, equal_nan=True):
            raise ValueError("The first eight augmented channels do not match the original dataset.")
        print("Original eight channels match the supplied unaugmented dataset.")

    print("Validation passed.")


if __name__ == "__main__":
    main()
