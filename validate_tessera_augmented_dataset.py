import argparse
import numpy as np


def infer_layout(x):
    if x.ndim != 5:
        raise ValueError(f"Expected a 5D array, got shape {x.shape}")

    if x.shape[1] >= 8 and x.shape[1] <= 200:
        return 1, 2  # [N, C, T, H, W]
    if x.shape[2] >= 8 and x.shape[2] <= 200:
        return 2, 1  # [N, T, C, H, W]
    raise ValueError(f"Could not infer channel/time layout from shape {x.shape}")


def main():
    parser = argparse.ArgumentParser(description="Validate TESSERA-augmented dataset concatenation.")
    parser.add_argument("path", type=str, help="Path to the saved .npy dataset.")
    args = parser.parse_args()

    x = np.load(args.path)
    print(f"Dataset shape: {x.shape}")

    channel_axis, time_axis = infer_layout(x)
    channels = x.shape[channel_axis]
    print(f"Channel count: {channels}")

    if channels < 8:
        raise ValueError(f"Unexpected channel count {channels}; expected at least 8 original channels.")

    if channels >= 136:
        print("TESSERA augmentation appears present: original channels + 128 appended.")
    else:
        print("No TESSERA augmentation detected at this point; channel count is still below 136.")

    # Inspect the last 128 channels only, which should be a repeated broadcast vector across time.
    tessera_start = channels - 128
    if channels < 128:
        print("Dataset has fewer than 128 channels; cannot inspect TESSERA block.")
        return

    if channel_axis == 1:
        tessera = x[:, tessera_start:, :, :, :]
        sample = tessera[0, :, :, 0, 0]
        # sample shape: (128, T)
        if sample.shape[1] > 1:
            all_same = np.allclose(sample, sample[:, :1].repeat(sample.shape[1], axis=1))
            print(f"Broadcast consistency across timesteps: {all_same}")
            if not all_same:
                print("Warning: TESSERA channels are not constant across time for this sample.")
        else:
            print("Only one timestep in sample; broadcast check is not meaningful.")
    else:
        tessera = x[:, :, tessera_start:, :, :]
        sample = tessera[0, :, :, 0, 0]
        # sample shape: (T, 128)
        if sample.shape[0] > 1:
            all_same = np.allclose(sample, sample[:1].repeat(sample.shape[0], axis=0))
            print(f"Broadcast consistency across timesteps: {all_same}")
            if not all_same:
                print("Warning: TESSERA channels are not constant across time for this sample.")
        else:
            print("Only one timestep in sample; broadcast check is not meaningful.")

    finite = np.isfinite(x).all()
    print(f"All values finite: {finite}")
    if not finite:
        print("Warning: NaN or inf values were found in the dataset.")


if __name__ == "__main__":
    main()
