"""
Andre Barle
CS7180 Project 3
December 12, 2025
Preprocess data from raw to a 3d log rgb histogram
"""

import os, random
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd


PROJECT_ROOT = Path(".").resolve()
# annotated cityscape data
CITY_ANNOT_ROOT = PROJECT_ROOT/"raw_data"/"archive"
CITY_IMG_ROOT = CITY_ANNOT_ROOT/"Cityscape Dataset"/"leftImg8bit"
CITY_GT_ROOT = CITY_ANNOT_ROOT/"Fine Annotations"/"gtFine"
# depth data cityscape
CITY_DEPTH_ROOT = PROJECT_ROOT/"raw_data"/"data"
# NYU depth data
NYU_ROOT = PROJECT_ROOT/"raw_data"/"nyu_data"/"data"
NYU_TRAIN_CSV = NYU_ROOT/"nyu2_train.csv"
NYU_TEST_CSV = NYU_ROOT/"nyu2_test.csv"
# Pix2Pix depth
P2P_ROOT = PROJECT_ROOT/"raw_data"/"pix2pix-depth"/"pix2pix-depth"
# list of cities for mask data
CITY_TRAIN_CITIES = ["bochum","bremen","cologne","darmstadt","dusseldorf","erfurt",
                     "hamburg","hanover","jena","krefeld","monchengladbach",
                     "strasbourg","stuttgart","tubingen","ulm","weimar","zurich"]
CITY_VAL_CITIES = ["frankfurt", "lindau", "munster"]
CITY_TEST_CITIES = ["berlin","bielefeld","bonn","leverkusen","mainz","munich"]

OUT_ROOT = PROJECT_ROOT/"processed_all"
IMG_SIZE = 256
HIST_BINS = 32
RANGE_STRAT = "percentile"

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def resize_srgb_array(arr, size=IMG_SIZE):
    # normalize to range [0,1], handles arrays
    arr = np.clip(arr, 0.0, 1.0)
    # resize to 256*256
    img = Image.fromarray((arr * 255.0).astype(np.uint8),
                          mode="RGB")
    img = img.resize((size, size), Image.BILINEAR)
    # convert back to float, ensure it is now resized
    out = np.asarray(img, dtype=np.float32) / 255.0
    return out

def load_rgb_resize(path, size=IMG_SIZE):
    # handles image files - performs the same as above, in resizing
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

# log chroma inversion
def srgb_to_linear(x):
    # uses the gamma transfer curve to get pseudo linear rgb
    a = 0.055
    thr = 0.04045
    # ensure values are normal
    x = np.clip(x, 0.0, 1.0)
    # transform
    return np.where(x <= thr, x/12.92, ((x + a)/(1 + a))**2.4)

def safe_log_rgb(img_lin, eps=1e-6):
    # avoids log 0 situation and returns three 2d arrays for RGB
    x = np.clip(img_lin, eps, None)
    return np.log(x[...,0]), np.log(x[...,1]), np.log(x[...,2])

def histogram3d_with_indices(logR, logG, logB,
                             bins=HIST_BINS,
                             range_strategy=RANGE_STRAT):
    # flatten to 1D each channel
    r, g, b = logR.ravel(), logG.ravel(), logB.ravel()
    # keep 98% of the data, throw away extreme outliers (noise)
    if range_strategy == "percentile":
        ranges = [(np.percentile(r,1), np.percentile(r,99)),
                  (np.percentile(g,1), np.percentile(g,99)),
                  (np.percentile(b,1), np.percentile(b,99))]
    # tried but did not work as fast
    elif range_strategy == "minmax":
        ranges = [(r.min(), r.max()), (g.min(), g.max()), (b.min(), b.max())]
    else:
        raise ValueError("Error in strategy")
    # form 3D histogram over log RGB
    H, edges = np.histogramdd(np.stack([r,g,b], 1), bins=bins, range=ranges)
    # get bin centers
    cx = 0.5*(edges[0][1:]+edges[0][:-1])
    cy = 0.5*(edges[1][1:]+edges[1][:-1])
    cz = 0.5*(edges[2][1:]+edges[2][:-1])
    # Compute bin indexes for pixels
    ir = np.digitize(r, edges[0]) - 1
    ig = np.digitize(g, edges[1]) - 1
    ib = np.digitize(b, edges[2]) - 1
    # if edge case (shares boundaries) then clip it to a valid range
    ir = np.clip(ir, 0, len(cx)-1)
    ig = np.clip(ig, 0, len(cy)-1)
    ib = np.clip(ib, 0, len(cz)-1)

    return H.astype(np.float32), (cx,cy,cz), edges, ranges, (ir, ig, ib)

def image_to_logchroma_binned(img_srgb, bins=HIST_BINS,
                              range_strategy=RANGE_STRAT):
    """
    Function: Uses helpers to make a reconstructable histogram representation
              of an image in log RGB space.
    Parameters: img_srgb, bins, range_strategy
    Returns:
        logrgb_q : (H,W,3) quantized log RGB from bin centers
        cx,cy,cz : (B,)
        bin_idx  : (H,W,3) ir,ig,ib for each pixel
    """
    # converts via helpers an sRGB image into log space
    img_srgb = np.clip(img_srgb.astype(np.float32), 0.0, 1.0)
    img_lin  = srgb_to_linear(img_srgb)
    logR, logG, logB = safe_log_rgb(img_lin)
    Hh, Ww = logR.shape
    # makes histogram
    _, (cx,cy,cz), _, _, (ir, ig, ib) = histogram3d_with_indices(
        logR, logG, logB, bins=bins, range_strategy=range_strategy)
    # reshape flattened to (H, W)
    ir_img = ir.reshape(Hh, Ww)
    ig_img = ig.reshape(Hh, Ww)
    ib_img = ib.reshape(Hh, Ww)
    # Use bin centers to get log pixel values into the center of the bin it should be in
    logR_q = cx[ir_img]
    logG_q = cy[ig_img]
    logB_q = cz[ib_img]
    logrgb_q = np.stack([logR_q, logG_q, logB_q], axis=2).astype(np.float32)
    # get the bin each pixel came from
    bin_idx = np.stack([ir_img, ig_img, ib_img], axis=2).astype(np.int16)
    # return the transformed image in log space, the bin index, and cy, cz, cx to reconstruct the image

    return logrgb_q, cx.astype(np.float32), cy.astype(np.float32), cz.astype(np.float32), bin_idx


def list_cityscapes_annotated(split, cities):
    """
    Function: gets all the images relavent to the data 
              using the directories and folders and names of cities
    parameters: split - train/val/test, cities - list of city names
    Returns image and label for each image
    """
    items = []
    for city in cities:
        img_dir = CITY_IMG_ROOT/split/city
        gt_dir  = CITY_GT_ROOT/split/city
        if not img_dir.exists() or not gt_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*_leftImg8bit.png")):
            stem = img_path.name.replace("_leftImg8bit.png", "")
            label = gt_dir / f"{stem}_gtFine_labelIds.png"
            if label.exists():
                items.append((img_path, label))
    return items

def resize_label_ids(label_arr, size=IMG_SIZE):
    # resize labels to fit image size
    label_arr = np.asarray(label_arr)
    pil = Image.fromarray(label_arr.astype(np.int32), mode="I")
    pil = pil.resize((size, size), Image.NEAREST)
    arr = np.asarray(pil, dtype=np.int32)
    return arr

def preprocess_cityscapes_masks():
    """
    Function: process the cityscapes labeled data
    """
    splits = {"train": (CITY_TRAIN_CITIES, "train"),
              "val": (CITY_VAL_CITIES, "val"),
              "test": (CITY_TEST_CITIES, "test"),}
    
    for split_name, (cities, split_dir) in splits.items():
        # get image and the semantic label masks
        pairs = list_cityscapes_annotated(split_dir, cities)
        # get output
        out_dir = OUT_ROOT/"mask"/"cityscapes"/split_name
        # verify directory
        ensure_dir(out_dir)
        # show progress
        print(f"[Masks] Cityscapes {split_name}: {len(pairs)} = {out_dir}")
        for idx, (img_path, label_path) in enumerate(pairs):
            # resize and scale
            rgb_srgb = load_rgb_resize(img_path, IMG_SIZE)
            # get semantic label and scale
            label_raw = np.array(Image.open(label_path))
            label = resize_label_ids(label_raw, IMG_SIZE)
            # pseudo log chroma histogram
            logrgb, cx, cy, cz, bin_idx = image_to_logchroma_binned(rgb_srgb)
            # format new name as numerical
            base = f"{idx:06d}"
            # compress and save all reconstruction data
            np.savez_compressed(out_dir / f"{base}.npz",
                                logrgb=logrgb.astype(np.float32),
                                bin_idx=bin_idx.astype(np.int16),
                                cx=cx, cy=cy, cz=cz,
                                label=label.astype(np.int32))
# depth data
def list_cityscapes_depth_split(split):
    # get images and ground truth depth
    img_dir = CITY_DEPTH_ROOT/split/"image"
    depth_dir = CITY_DEPTH_ROOT/split/"depth"
    # cityscape depth data is already in an array
    ids = sorted(p.stem for p in img_dir.glob("*.npy"))
    # get the pairs organized so they correspond
    pairs = []
    for _id in ids:
        img_p = img_dir / f"{_id}.npy"
        depth_p = depth_dir / f"{_id}.npy"
        if depth_p.exists():
            pairs.append((img_p, depth_p))
    # return GT depth and image as numpy files
    return pairs

def split_train_val(items, val_ratio=0.1, seed=42):
    # split randomly with seed into train and validation
    rng = random.Random(seed)
    idxs = list(range(len(items)))
    rng.shuffle(idxs)
    # 10% validation 90% training
    val_size = int(len(items)*val_ratio)
    val_idxs = set(idxs[:val_size])
    train, val = [], []
    for i, item in enumerate(items):
        (val if i in val_idxs else train).append(item)
    return train, val

def preprocess_cityscapes_depth():
    # we use the provided data which was labeled val as test data
    all_train = list_cityscapes_depth_split("train")
    # generate own val data from the original training set
    orig_val  = list_cityscapes_depth_split("val")

    # perform split
    train_pairs, val_pairs = split_train_val(all_train, val_ratio=0.1, seed=42)
    test_pairs = orig_val

    splits = {"train": train_pairs,
              "val": val_pairs,
              "test": test_pairs,}
    
    for split_name, pairs in splits.items():
        out_dir = OUT_ROOT/"depth"/"cityscapes"/split_name
        ensure_dir(out_dir)
        print(f"[Depth] Cityscapes {split_name}: {len(pairs)} = {out_dir}")
        for idx, (img_path, depth_path) in enumerate(pairs):
            rgb = np.load(img_path)
            # fix channel order if needed
            if rgb.ndim == 3 and rgb.shape[0] == 3:
                rgb = np.moveaxis(rgb, 0, 2)
            rgb = rgb.astype(np.float32)
            # normalize to [0,1] and resize
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            rgb = np.clip(rgb, 0.0, 1.0)
            rgb_srgb = resize_srgb_array(rgb, IMG_SIZE)
            # if depth is not monochrome, make it mono
            depth_raw = np.load(depth_path)
            depth = depth_raw.astype(np.float32)
            if depth.ndim == 3:
                depth = depth.mean(axis=2)
            dmin, dmax = depth.min(), depth.max()
            # get min and max depth values and normalize to [0,1]
            if dmax > dmin:
                depth01 = (depth - dmin) / (dmax - dmin)
            else:
                depth01 = np.zeros_like(depth, dtype=np.float32)
            # convert to PIL image and resize to fit rgb image
            d_img = Image.fromarray((depth01*255.0).astype(np.uint8), mode="L")
            d_img = d_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            depth01 = np.asarray(d_img, dtype=np.float32) / 255.0

            # pseudo log chroma historgram
            logrgb, cx, cy, cz, bin_idx = image_to_logchroma_binned(rgb_srgb)

            # save compressed image following numerical syntax
            base = f"{idx:06d}"
            np.savez_compressed(out_dir / f"{base}.npz",
                                logrgb=logrgb.astype(np.float32),
                                bin_idx=bin_idx.astype(np.int16),
                                cx=cx, cy=cy, cz=cz,
                                depth=depth01.astype(np.float32))
            
# only used for testing, depreciated for training
def load_nyu_pairs():
    # read into NYU csv data to get rgb and depth images
    df_train = pd.read_csv(NYU_TRAIN_CSV, header=None, sep=",", names=["rgb","depth"])
    df_test = pd.read_csv(NYU_TEST_CSV,  header=None, sep=",", names=["rgb","depth"])
    base = NYU_ROOT.parent

    # organize pairs for train and test
    def df_to_pairs(df):
        pairs = []
        for _, row in df.iterrows():
            rgb_p = base/str(row["rgb"])
            depth_p = base/str(row["depth"])
            if rgb_p.exists() and depth_p.exists():
                pairs.append((rgb_p, depth_p))
        return pairs

    # here we call the function to get the filepaths organized
    all_train = df_to_pairs(df_train)
    test_pairs = df_to_pairs(df_test)

    # make a random number and split the images randomly to get some validation images
    rng = random.Random(123)
    idxs = list(range(len(all_train)))
    rng.shuffle(idxs)
    val_size = int(len(idxs)*0.1)
    val_idx = set(idxs[:val_size])
    train_pairs, val_pairs = [], []
    for i, p in enumerate(all_train):
        (val_pairs if i in val_idx else train_pairs).append(p)

    # return train, val, and test
    return train_pairs, val_pairs, test_pairs

def preprocess_nyu_depth():
    # loads the pairs of data with depth gt and rgb image
    train_pairs, val_pairs, test_pairs = load_nyu_pairs()
    splits = {"train": train_pairs, "val": val_pairs, "test": test_pairs}
    # perform the split
    for split_name, pairs in splits.items():
        out_dir = OUT_ROOT/"depth"/"nyu"/split_name
        # verify output directory
        ensure_dir(out_dir)
        # show progress
        print(f"[Depth] NYU {split_name}: {len(pairs)} = {out_dir}")
        for idx, (rgb_path, depth_path) in enumerate(pairs):
            # load and resize
            rgb_srgb = load_rgb_resize(rgb_path, IMG_SIZE)
            # ensure we have a numpy format
            if depth_path.suffix == ".npy":
                depth_raw = np.load(depth_path)
            else:
                depth_raw = np.asarray(Image.open(depth_path))
            depth = depth_raw.astype(np.float32)
            # if depth is multidimensional make it 2D
            if depth.ndim == 3:
                depth = depth.mean(axis=2)
            dmin, dmax = depth.min(), depth.max()
            # normalize depth
            if dmax > dmin:
                depth01 = (depth - dmin) / (dmax - dmin)
            else:
                depth01 = np.zeros_like(depth, dtype=np.float32)
            # store as array
            d_img = Image.fromarray((depth01*255.0).astype(np.uint8), mode="L")
            d_img = d_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            depth01 = np.asarray(d_img, dtype=np.float32) / 255.0
            # convert to pseudo log RGB
            logrgb, cx, cy, cz, bin_idx = image_to_logchroma_binned(rgb_srgb)
            # save in numerical format and compress
            base = f"{idx:06d}"
            np.savez_compressed(out_dir / f"{base}.npz",
                                logrgb=logrgb.astype(np.float32),
                                bin_idx=bin_idx.astype(np.int16),
                                cx=cx, cy=cy, cz=cz,
                                depth=depth01.astype(np.float32))

# Pix2Pix data
# helper for splits
def list_pix2pix_split(split_name):
    split_dir = P2P_ROOT / split_name
    return sorted(split_dir.glob("*.png"))
# split data into training, val, testing
def preprocess_pix2pix_depth():
    splits = {"train": list_pix2pix_split("training"),
              "val":   list_pix2pix_split("validation"),
              "test":  list_pix2pix_split("testing"),}
    # split data
    for split_name, paths in splits.items():
        out_dir = OUT_ROOT/"depth"/"pix2pix"/split_name
        # verify directory
        ensure_dir(out_dir)
        # show progress
        print(f"[Depth] Pix2Pix {split_name}: {len(paths)} = {out_dir}")
        # need to physically divide the image into two halves, one half is the rgb, the other is depth gt
        for idx, comp_path in enumerate(paths):
            full = np.asarray(Image.open(comp_path).convert("RGB"))
            h, w, _ = full.shape
            mid = w // 2
            # split up the images and save them, scaled
            rgb = full[:, :mid, :].astype(np.float32) / 255.0
            depth_rgb = full[:, mid:, :].astype(np.float32) / 255.0
            depth_gray = depth_rgb.mean(axis=2)
            # resize image and depth
            rgb_srgb = resize_srgb_array(rgb, IMG_SIZE)
            d_img = Image.fromarray((depth_gray*255.0).astype(np.uint8), mode="L")
            d_img = d_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            depth01 = np.asarray(d_img, dtype=np.float32) / 255.0
            # reconstructable histogram
            logrgb, cx, cy, cz, bin_idx = image_to_logchroma_binned(rgb_srgb)
            # compress and save numerically
            base = f"{idx:06d}"
            np.savez_compressed(out_dir / f"{base}.npz",
                                logrgb=logrgb.astype(np.float32),
                                bin_idx=bin_idx.astype(np.int16),
                                cx=cx, cy=cy, cz=cz,
                                depth=depth01.astype(np.float32))
# main
if __name__ == "__main__":
    preprocess_cityscapes_masks()
    preprocess_cityscapes_depth()
    preprocess_nyu_depth()
    preprocess_pix2pix_depth()
    print("All datasets preprocessed into processed_all/")
