"""
Andre Barle
CS7180 Project 3
December 12, 2025
Project demo reads in a saved model and 
prepared data and shows summary stats and performs predictions
"""

import os
import time
from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from final_model import (SharedTransformerDepth,
                         default_device,
                         reconstruct_srgb_from_logrgb_tensor,)

from final_preprocess_data import (IMG_SIZE,
                                   load_rgb_resize,
                                   image_to_logchroma_binned,)

# instatiate the classes from semantic labeled data
def infer_classes(processed_root: Path) -> int:
    # get the classes from the data
    mask_train_dir = processed_root/"mask"/"cityscapes"/"train"
    # order by filename
    files = sorted(mask_train_dir.glob("*.npz"))
    # show if error
    if not files:
        raise RuntimeError(f"No Cityscapes mask files found in {mask_train_dir}")
    # get all the labels, start with 0, update to largest and seek until all labels are found
    max_label = 0
    for f in files:
        z = np.load(f)
        max_label = max(max_label, int(np.max(z["label"])))
    return max_label + 1

# test to verify that the model learned depth regardless of orientiation (it does)
def orientation_augmentation(logrgb: np.ndarray, depth: np.ndarray, idx: int):
    # apply random flips to the first 10 images being tested
    if idx >= 10:
        return logrgb, depth, "none"
    mode = idx % 4
    if mode == 0:
        # horizontal flip
        logrgb_aug = np.flip(logrgb, axis=1)
        depth_aug = np.flip(depth, axis=1)
        aug_name = "hflip"
    elif mode == 1:
        # vertical flip
        logrgb_aug = np.flip(logrgb, axis=0)
        depth_aug = np.flip(depth, axis=0)
        aug_name = "vflip"
    elif mode == 2:
        # rotate 90
        logrgb_aug = np.rot90(logrgb, k=1, axes=(0, 1))
        depth_aug = np.rot90(depth, k=1, axes=(0, 1))
        aug_name = "rot90"
    else:
        # rotate 270
        logrgb_aug = np.rot90(logrgb, k=3, axes=(0, 1))
        depth_aug = np.rot90(depth, k=3, axes=(0, 1))
        aug_name = "rot270"

    return logrgb_aug, depth_aug, aug_name

# tool for me to put in a personal image and have the model evaluate it
def find_my_image(script_dir: Path) -> Path:
    # go to the directory
    my_dir = script_dir/"my_images"
    # raise error
    if not my_dir.is_dir():
        raise RuntimeError(f"not found")

    # image extensions
    exts = [".heic", ".HEIC", ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

    # lets you do multiple images if they fit the format
    candidates = sorted(p for p in my_dir.iterdir() if p.is_file() and p.suffix in exts)
    if not candidates:
        raise RuntimeError(f"no valid images found")
    return candidates[0]
# get timing of prediction
def sync_device(device: torch.device):
    # sync up device
    if device.type == "cuda":
        torch.cuda.synchronize()
    # I use mps but cuda is there if you have it
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()

# predictions
def main():
    # arguments to navigate the demo, explained in readme
    ap = argparse.ArgumentParser(description="Run depth predictions")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/tri_transformer_depth_logchroma_ep151.pth",
                    help="get to model checkpoint",)
    ap.add_argument("--data-root", type=str, default="processed_all",
                    help="root for data",)
    ap.add_argument("--dataset", type=str, default="nyu", choices=["nyu", "cityscapes", "pix2pix"],
                    help="select a dataset",)
    ap.add_argument("--num-images", type=int, default=10,
                    help="how many images do you want to predict",)
    ap.add_argument("--save-dir", type=str, default="predictions_test", 
                    help="save output",)
    ap.add_argument("--use-my-image", action="store_true", 
                    help="predict on the custom image you choose instead of the datasets",
    )
    ap.add_argument( "--eval-metrics", action="store_true", help=("get metrics for the model"),)
    args = ap.parse_args()

    # set device
    device = default_device()
    print("Device:", device)
    # get directories
    script_dir = Path(__file__).resolve().parent
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # get classes from segmented labeled data
    data_root = Path(args.data_root).resolve()
    num_classes = infer_classes(data_root)
    print("Inferred num_classes (Cityscapes):", num_classes)

    # instantiate model and load it
    model = SharedTransformerDepth(num_classes=num_classes).to(device)
    ckpt_path = Path(args.checkpoint)
    # if error
    if not ckpt_path.is_file():
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}")
    # load it
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # for a custom image
    if args.use_my_image:
        img_path = find_my_image(script_dir)
        print(f"Using custom image: {img_path}")

        # load and resize
        rgb_srgb = load_rgb_resize(img_path, IMG_SIZE)

        # Convert to log rgb
        logrgb, cx, cy, cz, bin_idx = image_to_logchroma_binned(rgb_srgb)

        # ensure contiguous image or torch wont work
        logrgb = np.ascontiguousarray(logrgb)

        # get into a tensor
        x = torch.from_numpy(np.moveaxis(logrgb, 2, 0)).unsqueeze(0).to(device)

        # timing measurement
        with torch.no_grad():
            # start the prediction
            _ = model.forward_depth(x)
            sync_device(device)
            t0 = time.perf_counter()
            _, depth_refined = model.forward_depth(x)
            sync_device(device)
            t1 = time.perf_counter()
        
        # measure time elapsed
        elapsed = t1 - t0
        if elapsed > 0:
            imgs_per_sec = 1.0 / elapsed
            preds_30s = imgs_per_sec * 30.0
        else:
            imgs_per_sec = float("inf")
            preds_30s = float("inf")
        # get summary statistics of performance
        print("\nInference speed")
        print(f"Single forward_depth time: {elapsed*1000} ms")
        print(f"Images per second: {imgs_per_sec}")
        print(f"Estimated predictions in 30 s: {preds_30s}")

        # reconstruct approx image from log rgb histogram
        rgb_recon = reconstruct_srgb_from_logrgb_tensor(x[0].cpu())
        pr_d = depth_refined[0, 0].cpu().numpy()

        # plot predictions
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(rgb_srgb)
        # show base image
        axs[0].set_title("Input RGB (resized)")
        axs[0].axis("off")
        # predicted depth
        im2 = axs[1].imshow(pr_d, cmap="viridis")
        axs[1].set_title("Predicted Depth (refined)")
        axs[1].axis("off")
        fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
        # save the prediction
        out_name = save_dir / f"my_image_{img_path.stem}.png"
        plt.tight_layout()
        plt.savefig(out_name, dpi=150)
        plt.close(fig)

        print(f"Saved prediction to: {out_name}")
        return

    # dataset testing
    depth_test_dir = data_root/"depth"/args.dataset/"test"
    # get the files
    all_files = sorted(depth_test_dir.glob("*.npz"))
    # if error
    if not all_files:
        raise RuntimeError(f"No test .npz files found in {depth_test_dir}")
    # show evaluation metrics
    if args.eval_metrics:
        files = all_files
        print(f"Found {len(files)} test files in {depth_test_dir}.")
    else:
        files = all_files[: args.num_images]
        print(f"Found {len(all_files)} test files in {depth_test_dir};"
              f"using first {len(files)} for visualization.")

    # Count metrics if eval is on
    total_pixels = 0
    # MAE
    sum_abs = 0.0
    # MSE/RMSE
    sum_sq = 0.0
    sum_rel = 0.0
    correct_delta1 = 0
    total_delta = 0

    # timing of predictions
    total_infer_time = 0.0
    infer_count = 0

    # iterate over the test set
    for i, f in enumerate(files):
        z = np.load(f)
        logrgb = z["logrgb"].astype(np.float32)
        depth_gt = z["depth"].astype(np.float32)

        # dont augment the test data for this estimation
        if args.eval_metrics:
            aug_name = "none"
        else:
            logrgb, depth_gt, aug_name = orientation_augmentation(logrgb, depth_gt, i)

        # ensure arrays are contiguous
        logrgb = np.ascontiguousarray(logrgb)
        depth_gt = np.ascontiguousarray(depth_gt)

        # get tensors
        x = torch.from_numpy(np.moveaxis(logrgb, 2, 0)).unsqueeze(0).to(device)  # (1,3,H,W)
        depth_t = torch.from_numpy(depth_gt[None, None, ...]).to(device)         # (1,1,H,W)

        # time predictions
        with torch.no_grad():
            # start predictions
            if infer_count == 0:
                _ = model.forward_depth(x)
                sync_device(device)
            # count the time
            t0 = time.perf_counter()
            _, depth_refined = model.forward_depth(x)
            sync_device(device)
            t1 = time.perf_counter()

        # get the time measurement
        total_infer_time += (t1 - t0)
        infer_count += 1

        # reconstruct srgb from logrgb
        rgb_recon = reconstruct_srgb_from_logrgb_tensor(x[0].cpu())
        gt_d = depth_t[0, 0].cpu().numpy()
        pr_d = depth_refined[0, 0].cpu().numpy()

        # get matrics for test
        if args.eval_metrics:
            # ensure depths are within normal range of values
            valid = np.isfinite(gt_d) & (gt_d > 0)
            # ground truth values
            gt_valid = gt_d[valid]
            # predicted depth values
            pr_valid = pr_d[valid]
            # number of valid pixels
            n_valid = gt_valid.size

            if n_valid > 0:
                # per pixel error
                diff = pr_valid - gt_valid
                abs_diff = np.abs(diff)
                # accumulate pixels
                total_pixels += n_valid
                # sum of predicted - gt for pixels
                sum_abs += abs_diff.sum()
                # squared error per pixel
                sum_sq += (diff ** 2).sum()
                # get relative error
                sum_rel += (abs_diff / (gt_valid + 1e-6)).sum()

                # get threshold for how close pred and gt is
                ratio = np.maximum(pr_valid / (gt_valid + 1e-6),
                                   gt_valid / (pr_valid + 1e-6),)
                # true when prediction is in the threshold
                correct_delta1 += (ratio < 1.25).sum()
                # total number of valid pixels
                total_delta += n_valid

        # visualize n images
        if i < args.num_images:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(rgb_recon)
            axs[0].set_title(f"Reconstructed RGB\n(aug: {aug_name})")
            axs[0].axis("off")

            im1 = axs[1].imshow(gt_d, cmap="viridis")
            axs[1].set_title("GT Depth")
            axs[1].axis("off")
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

            im2 = axs[2].imshow(pr_d, cmap="viridis")
            axs[2].set_title("Predicted Depth (refined)")
            axs[2].axis("off")
            fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

            fname = save_dir / f"test_{i}_{aug_name}.png"
            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"[{i+1}/{len(files)}] Saved {fname}")
        else:
            # no visualization, continue
            print(f"[{i+1}/{len(files)}] Processed {f.name}")

    # final metrics
    if args.eval_metrics and total_pixels > 0:
        mae = sum_abs / total_pixels
        mse = sum_sq / total_pixels
        rmse = np.sqrt(mse)
        abs_rel = sum_rel / total_pixels
        acc_delta1 = correct_delta1 / max(total_delta, 1)

        print("\nFull test-set metrics")
        print(f"Total valid pixels: {total_pixels}")
        print(f"L1 (MAE): {mae}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"Abs-Rel: {abs_rel}")
        print(f"threshold accuracy: {acc_delta1*100}%")

    # prediction speed
    if infer_count > 0 and total_infer_time > 0:
        avg_time = total_infer_time / infer_count
        imgs_per_sec = infer_count / total_infer_time
        preds_30s = imgs_per_sec * 30.0

        print("\nPrediction speed")
        print(f"Images processed: {infer_count}")
        print(f"Total inference time: {total_infer_time} s")
        print(f"Average time per image: {avg_time*1000} ms")
        print(f"Images per second: {imgs_per_sec}")
        print(f"Estimated preds in 30 s: {preds_30s}")
    else:
        print("\nFailed")

    print("predictions saved to:", save_dir)

if __name__ == "__main__":
    main()
