"""
Andre Barle
CS7180 Project 3
December 12, 2025
Train the model and save checkpoints each epoch
"""

import os, time, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.models as models

PROJECT_ROOT = Path(".").resolve()
DATA_ROOT    = PROJECT_ROOT/"processed_all"
IMG_SIZE     = 256

class StepETAMeter:
    def __init__(self, total_units):
        # batches expected to process
        self.total = max(1, int(total_units))
        # completed so far
        self.done = 0
        # start time
        self.t0 = time.perf_counter()
    # increment progress
    def update(self, n=1): self.done += int(n)
    @property
    # seconds since start
    def elapsed(self): return time.perf_counter() - self.t0
    @property 
    # throughput: batch per second
    def rate(self): return self.done / self.elapsed if self.elapsed > 0 else 0.0
    @property
    # get eta
    def eta_seconds(self):
        left = max(0, self.total - self.done)
        return left / self.rate if self.rate > 0 else float("inf")

class EpochETAMeter:
    def __init__(self, total_epochs, ema_beta=0.3):
        # total epochs to run
        self.total_epochs = total_epochs
        # weight for current epoch duration as exponential moving average
        self.ema_beta = ema_beta
        # epochs completed
        self.epoch_idx = 0
        # smoothed estimation in seconds
        self._ema = None
        # start for current epoch
        self._epoch_t0 = None
    # start time
    def start_epoch(self): self._epoch_t0 = time.perf_counter()
    def end_epoch(self):
        # seconds epoch ran
        dt = time.perf_counter() - self._epoch_t0
        # update ema of duration
        self._ema = dt if self._ema is None else self.ema_beta * dt + (1-self.ema_beta)*self._ema
        # increment count of epochs
        self.epoch_idx += 1
        # return time
        return dt
    # if no epoch finished yet
    @property
    def overall_eta_seconds(self):
        if self._ema is None: return float("inf")
        remaining = max(0, self.total_epochs - self.epoch_idx)
        return remaining * self._ema
    
# config should work for any OS to resolve torch cuda use
def default_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_augment_depth(x, depth, img_size=IMG_SIZE, prob=0.5):
    """
    Function: augments the data by randomly rotating, flipping,
              and rescaling the data to generate more samples and
              to destroy vertical image bias for depth
    Parameters: x: (3,H,W) tensor
                depth: (1,H,W) tensor
                img_size: image size
                prob: probability of alteration
    """
    if random.random() > prob:
        return x, depth

    _, H, W = x.shape
    # random rotations [0,90,180,270]
    k = random.randint(0, 3)
    if k > 0:
        x = torch.rot90(x, k, dims=(1, 2))
        depth = torch.rot90(depth, k, dims=(1, 2))
    # random crop and resize
    crop_scale = 0.8
    crop_h = int(H * crop_scale)
    crop_w = int(W * crop_scale)
    if crop_h < H and crop_w < W:
        # random offsets of images and depth
        top  = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)
        x_crop = x[:, top:top+crop_h, left:left+crop_w]
        depth_crop = depth[:, top:top+crop_h, left:left+crop_w]
        # unsqueeze and resize with interpolation, then squeeze back down
        x = F.interpolate(x_crop.unsqueeze(0), size=(img_size, img_size),
                              mode="bilinear", align_corners=False).squeeze(0)
        depth = F.interpolate(depth_crop.unsqueeze(0), size=(img_size, img_size),
                              mode="bilinear", align_corners=False).squeeze(0)

    return x, depth


def random_augment_mask(x, label, img_size=IMG_SIZE, prob=0.5):
    """
    Function: augments the data by randomly rotating, flipping,
              and rescaling the data to generate more samples and
              to destroy vertical image bias for depth
    Parameters: x: (3,H,W) tensor
                label: (H,W) tensor with semantic IDs
                img_size: image size
                prob: probability of alteration
    """
    if random.random() > prob:
        return x, label

    H, W = label.shape
    # random rotations [0,90,180,270]
    k = random.randint(0, 3)
    if k > 0:
        x = torch.rot90(x, k, dims=(1, 2))
        label = torch.rot90(label, k, dims=(0, 1))
    # random crop and resize
    crop_scale = 0.8
    crop_h = int(H * crop_scale)
    crop_w = int(W * crop_scale)
    if crop_h < H and crop_w < W:
        # random offsets of images and depth
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)
        x_crop = x[:,top:top+crop_h, left:left+crop_w]
        label_crop = label[top:top+crop_h, left:left+crop_w]

        x = F.interpolate(x_crop.unsqueeze(0), size=(img_size, img_size),
                          mode="bilinear", align_corners=False).squeeze(0)

        # unsqueeze and resize with interpolation, then squeeze back down
        label_f = label_crop.unsqueeze(0).unsqueeze(0).float()
        # this time we interpolate nearest to preserve the class IDs
        label_r = F.interpolate(label_f, size=(img_size, img_size),
                                mode="nearest").squeeze(0).squeeze(0)
        label = label_r.long()

    return x, label

# Data loaders
class DepthNPZDataset(Dataset):
    def __init__(self, dataset_name: str, split: str, augment: bool = False):
        # directory to read
        self.base_dir = DATA_ROOT/"depth"/dataset_name/split
        self.files = sorted(self.base_dir.glob("*.npz"))
        # raise error if no file
        if not self.files:
            raise RuntimeError(f"No .npz files in {self.base_dir}")
        sample = np.load(self.files[0])
        # confirm it loaded
        print(f"[DepthNPZDataset] {dataset_name}/{split}: {len(self.files)} samples")
        print("logrgb:", sample["logrgb"].shape, "depth:", sample["depth"].shape)
        self.augment = augment
    # length
    def __len__(self):
        return len(self.files)
    # get file and load logrgb and depth
    def __getitem__(self, idx):
        # load data
        data = np.load(self.files[idx])
        logrgb = data["logrgb"].astype(np.float32)
        depth = data["depth"].astype(np.float32)
        # reshape from last axis to first and add a channel to depth
        x = torch.from_numpy(np.moveaxis(logrgb, 2, 0))
        dep = torch.from_numpy(depth[None, ...])
        # augment data
        if self.augment:
            x, dep = random_augment_depth(x, dep, img_size=IMG_SIZE, prob=0.5)

        return x, dep


class CityscapesMaskNPZDataset(Dataset):
    def __init__(self, split: str, augment: bool = False):
        # directory to read
        self.base_dir = DATA_ROOT/"mask"/"cityscapes"/split
        self.files = sorted(self.base_dir.glob("*.npz"))
        # raise error if no file
        if not self.files:
            raise RuntimeError(f"No .npz files in {self.base_dir}")
        sample = np.load(self.files[0])
        # confirm it loaded
        print(f"[CityscapesMaskNPZDataset] {split}: {len(self.files)} samples")
        print("  logrgb:", sample["logrgb"].shape, "label:", sample["label"].shape)
        self.augment = augment
    # length
    def __len__(self):
        return len(self.files)
    # get file and load logrgb and depth
    def __getitem__(self, idx):
        # load data
        data = np.load(self.files[idx])
        logrgb = data["logrgb"].astype(np.float32)
        label = data["label"].astype(np.int32)
        # reshape from last axis to first and add a channel to depth
        x = torch.from_numpy(np.moveaxis(logrgb, 2, 0))
        label = torch.from_numpy(label)
        # augment data
        if self.augment:
            x, label = random_augment_mask(x, label.long(), img_size=IMG_SIZE, prob=0.5)

        return x, label

class ConvEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, pdrop=0.1):
        super().__init__()
        def block(ic, oc):
            # build encoder 3 input channels, dropout and batch normalization
            return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1),
                                 nn.BatchNorm2d(oc),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(pdrop),)
        self.pool = nn.MaxPool2d(2, 2)
        # 256x256
        self.b1 = block(in_ch, base_ch)
        # 128x128
        self.b2 = block(base_ch, base_ch*2)
        # 64x64
        self.b3 = block(base_ch*2, base_ch*4)

    # forward pass
    def forward(self, x):
        # 256x256
        x1 = self.b1(x)
        p1 = self.pool(x1)
        # 128x128
        x2 = self.b2(p1)
        p2 = self.pool(x2)
        # 64x64
        x3 = self.b3(p2)

        return x1, x2, x3


class DepthDecoder(nn.Module):
    def __init__(self, token_dim=256, enc_ch2=64, enc_ch1=32,
                 base_ch=64, img_size=IMG_SIZE):
        super().__init__()
        self.img_size = img_size
        # project feature map to base channels of decoder
        self.conv_in = nn.Conv2d(token_dim, base_ch*4, 1)
        # upsample from 64*64 to 128*128
        self.up1 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        # skip and map to encoder channels
        self.skip2_conv = nn.Conv2d(enc_ch2, base_ch*2, 1)
        # convolve over skip and features
        self.dec2 = nn.Sequential(nn.Conv2d(base_ch*4, base_ch*2, 3, padding=1),
                                  nn.BatchNorm2d(base_ch*2),
                                  nn.ReLU(inplace=True),)

        # upsample from 128*128 to 256*256
        self.up2 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        # skip and map to encoder channels
        self.skip1_conv = nn.Conv2d(enc_ch1, base_ch, 1)
        # convolve over skip and features
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )

        # output to depth channel after convolution
        self.out = nn.Conv2d(base_ch, 1, 1)

    def forward(self, feat_tr, skip2, skip1, Hs, Ws):
        # transformer output to decoder
        x = self.conv_in(feat_tr)
        # upsample and merge mid-res features from encoder
        x = self.up1(x)
        s2 = self.skip2_conv(skip2)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x)
        # upsample and merge high-res features from encoder
        x = self.up2(x)
        s1 = self.skip1_conv(skip1)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1(x)
        # check image size is correct
        x = F.interpolate(x, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)
        depth = torch.sigmoid(self.out(x))

        return depth


class MaskDecoder(nn.Module):
    def __init__(self, token_dim=256, enc_ch2=64, enc_ch1=32,
                 num_classes=20, base_ch=64, img_size=IMG_SIZE):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        # project feature map to base channels of decoder
        self.conv_in = nn.Conv2d(token_dim, base_ch*4, 1)

        # upsample from 64*64 to 128*128
        self.up1 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        # skip and map to encoder channels
        self.skip2_conv = nn.Conv2d(enc_ch2, base_ch*2, 1)
        # convolve over skip and features
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*2, 3, padding=1),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(inplace=True),
        )

        # upsample from 128*128 to 256*256
        self.up2 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        # skip and map to encoder channels
        self.skip1_conv = nn.Conv2d(enc_ch1, base_ch, 1)
        # convolve over skip and features
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
       
        self.out = nn.Conv2d(base_ch, num_classes, 1)

    def forward(self, feat_tr, skip2, skip1, Hs, Ws):
        x = self.conv_in(feat_tr)
        # 128*128
        x = self.up1(x)
        # merge with skip
        s2 = self.skip2_conv(skip2)
        x = torch.cat([x, s2], dim=1)
        # fused transformer output with encoder
        x = self.dec2(x)

        # 256x256
        x = self.up2(x)
        # merge with skip
        s1 = self.skip1_conv(skip1)
        x = torch.cat([x, s1], dim=1)
        # fused transformer output with encoder
        x = self.dec1(x)

        # ensure correct image size
        x = F.interpolate(x, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)
        logits = self.out(x)
        # not softmax, just raw
        return logits

class DepthRefineHead(nn.Module):
    def __init__(self, num_classes, base_ch=32):
        super().__init__()
        # merge output from the classifier with the depth predictor
        in_ch = 1 + num_classes
        # 2 layer one output layer CNN for refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, 3, padding=1),
        )

    def forward(self, depth_coarse, mask_logits):
        # probabilities computed via softmax
        mask_probs = F.softmax(mask_logits, dim=1)
        # merge with depth info
        x = torch.cat([depth_coarse, mask_probs], dim=1)
        # predict residual and add to depth
        residual = self.refine(x)
        # output a corrected depth
        depth_refined = torch.sigmoid(depth_coarse + residual)
        return depth_refined

class SharedTransformerDepth(nn.Module):
    # get encoder base channels and dropout
    def __init__(self, num_classes, base_ch=32,
                 d_model=256, nhead=4,num_layers=2,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = ConvEncoder(in_ch=3, base_ch=base_ch, pdrop=dropout)

        # 2 max pools reduce the size to 64
        self.Hs = IMG_SIZE // 4
        self.Ws = IMG_SIZE // 4
        # 3 channels
        feat_ch = base_ch * 4

        # project features to get d_model channels
        self.proj = nn.Conv2d(feat_ch, d_model, 1)

        # encoder gives tokens (B,N,D)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Decoders instantiated
        self.depth_decoder = DepthDecoder(
            token_dim=d_model,
            enc_ch2=base_ch*2,
            enc_ch1=base_ch,
            base_ch=64,
            img_size=IMG_SIZE)
        
        self.mask_decoder  = MaskDecoder(
            token_dim=d_model,
            enc_ch2=base_ch*2,
            enc_ch1=base_ch,
            num_classes=num_classes,
            base_ch=64,
            img_size=IMG_SIZE)

        # refinement head instantiated
        self.refine_head = DepthRefineHead(num_classes=num_classes, base_ch=32)

    def _encode(self, x):
        # return fine, medium, and coarse feature maps
        x1, x2, x3 = self.encoder(x)
        feat = self.proj(x3)
        # batch size, channels, height, width
        B, D, Hs, Ws = feat.shape
        # flatten and run transformer
        tokens = feat.view(B, D, Hs*Ws).permute(0,2,1)
        tokens = self.transformer(tokens)
        # reshape back to 2D features
        feat_tr = tokens.view(B, Hs, Ws, D).permute(0,3,1,2).contiguous()  # (B,D,64,64)
        return feat_tr, x2, x1, Hs, Ws

    def forward_depth(self, logrgb):
        # encoder, transformer, reshape
        feat_tr, x2, x1, Hs, Ws = self._encode(logrgb)
        # get depth prediction
        depth_coarse = self.depth_decoder(feat_tr, x2, x1, Hs, Ws)
        # get semantic predictions for refinement
        mask_logits = self.mask_decoder(feat_tr, x2, x1, Hs, Ws)
        # get output from refinement head
        depth_refined = self.refine_head(depth_coarse, mask_logits)
        # get final depths, refined and not refined
        return depth_coarse, depth_refined

    def forward_mask(self, logrgb):
        # encoder, transformer, reshape
        feat_tr, x2, x1, Hs, Ws = self._encode(logrgb)
        # upsampling and predicting probabilities of masks
        logits = self.mask_decoder(feat_tr, x2, x1, Hs, Ws)
        return logits

# perceptual loss
class VGG16Perceptual(nn.Module):
    def __init__(self):
        super().__init__()
        # load the weights for perceptual loss
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # get those features
        features = list(vgg.features.children())
        # feature extractor for perceptual loss and frozen weights
        self.slice = nn.Sequential(*features[:16])
        for p in self.slice.parameters():
            p.requires_grad = False

        # normalization constants for VGG16
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        # resize to VGG training size
        x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
        # normalize channels
        x = (x - self.mean) / self.std
        # pass through frozen VGG
        return self.slice(x)


def perceptual_loss_vgg(vgg, pred_depth, gt_depth):
    # 3 channel greyscale rgb image conversion
    pred_3 = pred_depth.repeat(1, 3, 1, 1)
    gt_3 = gt_depth.repeat(1, 3, 1, 1)
    # normalize
    pred_3 = torch.clamp(pred_3, 0.0, 1.0)
    gt_3 = torch.clamp(gt_3,   0.0, 1.0)
    # pass through VGG module 
    feat_pred = vgg(pred_3)
    feat_gt = vgg(gt_3)
    # get the L1 loss between feature maps
    return F.l1_loss(feat_pred, feat_gt)

def _linear_to_srgb_np(x):
    # convert from linear to srgb using the standard formula
    a = 0.055
    # normalize
    x = np.clip(x, 0.0, 1.0)
    # gamma encoding reversal
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        (1 + a) * np.power(x, 1/2.4) - a
    )

def reconstruct_srgb_from_logrgb_tensor(logrgb_tensor):
    # break gradient, tensor is on cpu, convert to numpy and transpose
    logrgb = logrgb_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    # get each channel into a 2D array
    logR, logG, logB = logrgb[...,0], logrgb[...,1], logrgb[...,2]
    # reverse log
    R_lin = np.exp(logR)
    G_lin = np.exp(logG)
    B_lin = np.exp(logB)
    # stack them back to one array
    lin = np.stack([R_lin, G_lin, B_lin], axis=2).astype(np.float32)
    # convert back to srgb
    srgb = _linear_to_srgb_np(lin)
    srgb = np.clip(srgb, 0.0, 1.0).astype(np.float32)
    return srgb

# eval
# show the sample data for depth
def show_depth_batch(loader):
    x, depth = next(iter(loader))
    # reconstruct rgb from logrgb
    rgb_recon = reconstruct_srgb_from_logrgb_tensor(x[0])
    dep  = depth[0,0].numpy()
    fig, axs = plt.subplots(1,2,figsize=(8,4))
    axs[0].imshow(rgb_recon); axs[0].set_title("Reconstructed RGB (depth dataset)"); axs[0].axis("off")
    axs[1].imshow(dep, cmap="viridis");  axs[1].set_title("Depth GT"); axs[1].axis("off")
    plt.tight_layout(); plt.show()

# show the sample data for semantic masks
def show_mask_batch(loader):
    x, label = next(iter(loader))
    rgb_recon = reconstruct_srgb_from_logrgb_tensor(x[0])
    lab  = label[0].numpy()
    fig, axs = plt.subplots(1,2,figsize=(8,4))
    axs[0].imshow(rgb_recon); axs[0].set_title("Reconstructed RGB (mask dataset)"); axs[0].axis("off")
    axs[1].imshow(lab, cmap="tab20", interpolation="nearest"); axs[1].set_title("Label IDs"); axs[1].axis("off")
    plt.tight_layout(); plt.show()

# compute L1 error over the data
@torch.no_grad()
def evaluate_depth(model, loader, device):
    # evaluate
    model.eval()
    # loss counter
    total_l1, n = 0.0, 0
    # loop over all batches in loader
    for x, depth in loader:
        x, depth = x.to(device), depth.to(device)
        _, depth_refined = model.forward_depth(x)
        # total loss for depth prediction (intermediate step)
        loss = F.l1_loss(depth_refined, depth)
        total_l1 += loss.item() * x.size(0)
        n += x.size(0)
    return total_l1 / max(1,n)

@torch.no_grad()
def evaluate_mask(model, loader, device):
    # evaluate
    model.eval()
    # loss counter
    ce = nn.CrossEntropyLoss()
    total_ce, correct, total = 0.0, 0, 0
    # loop over all batches in loader
    for x, label in loader:
        x, label = x.to(device), label.to(device)
        logits = model.forward_mask(x)
        # total loss for segmentation prediction (intermediate step)
        loss_ce = ce(logits, label)
        # add up cross entropy loss
        total_ce += loss_ce.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        # number of pixels in label across the batch
        total += label.numel()
    return total_ce / max(1, len(loader.dataset)), correct / max(1, total)

@torch.no_grad()
def visualize_predictions(model, depth_loader, mask_loader, device):
    model.eval()
    # depth image
    x_d, depth_gt = next(iter(depth_loader))
    x_d, depth_gt = x_d.to(device), depth_gt.to(device)
    depth_coarse, depth_refined = model.forward_depth(x_d)

    # reconstructed rgb from logrgb
    rgb_recon = reconstruct_srgb_from_logrgb_tensor(x_d[0])

    gt_d    = depth_gt[0,0].cpu().numpy()
    dr_d    = depth_refined[0,0].cpu().numpy()

    # show predictions
    fig, axs = plt.subplots(1,3,figsize=(12,4))
    axs[0].imshow(rgb_recon); axs[0].set_title("Reconstructed RGB"); axs[0].axis("off")
    axs[1].imshow(gt_d, cmap="viridis"); axs[1].set_title("GT Depth"); axs[1].axis("off")
    axs[2].imshow(dr_d, cmap="viridis"); axs[2].set_title("Refined Pred Depth"); axs[2].axis("off")
    plt.tight_layout(); plt.show()

    # semantic label image
    x_m, label_gt = next(iter(mask_loader))
    x_m, label_gt = x_m.to(device), label_gt.to(device)
    logits = model.forward_mask(x_m)
    pred_mask = logits.argmax(dim=1)[0].cpu().numpy()
    rgb_recon_m = reconstruct_srgb_from_logrgb_tensor(x_m[0])
    lab_m     = label_gt[0].cpu().numpy()

    # show predictions
    fig, axs = plt.subplots(1,3,figsize=(12,4))
    axs[0].imshow(rgb_recon_m); axs[0].set_title("Reconstructed RGB (mask)"); axs[0].axis("off")
    axs[1].imshow(lab_m, cmap="tab20", interpolation="nearest"); axs[1].set_title("GT Mask"); axs[1].axis("off")
    axs[2].imshow(pred_mask, cmap="tab20", interpolation="nearest"); axs[2].set_title("Pred Mask"); axs[2].axis("off")
    plt.tight_layout(); plt.show()

# train
def train_tri_transformer(epochs=10, batch_size=4, lr=1e-4,
                          w_l1=0.5, w_perc=0.5,):
    device = default_device()
    print("Device:", device)

    # depth datasets without NYU
    depth_train = ConcatDataset([DepthNPZDataset("cityscapes","train", augment=True),
                                 DepthNPZDataset("pix2pix","train",    augment=True),])
    depth_val   = ConcatDataset([DepthNPZDataset("cityscapes","val", augment=False),
                                 DepthNPZDataset("pix2pix","val",    augment=False),])
    # load the depth data
    depth_train_loader = DataLoader(depth_train, batch_size=batch_size,
                                    shuffle=True, num_workers=0)
    depth_val_loader = DataLoader(depth_val, batch_size=batch_size,
                                  shuffle=False, num_workers=0)
    # semantic label data
    mask_train_ds = CityscapesMaskNPZDataset("train", augment=True)
    mask_val_ds = CityscapesMaskNPZDataset("val", augment=False)
    num_classes = int(max(np.max(np.load(f)["label"]) for f in mask_train_ds.files)) + 1
    # load the semantic labeled data
    mask_train_loader = DataLoader(mask_train_ds, batch_size=batch_size,
                                   shuffle=True, num_workers=0)
    mask_val_loader   = DataLoader(mask_val_ds,   batch_size=batch_size,
                                   shuffle=False, num_workers=0)

    # let me see examples
    show_depth_batch(depth_train_loader)
    show_mask_batch(mask_train_loader)

    # model run
    model = SharedTransformerDepth(num_classes=num_classes).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    ce    = nn.CrossEntropyLoss()
    # instatiate VGG
    vgg_perc = VGG16Perceptual().to(device)
    vgg_perc.eval()
    # timer
    epoch_meter = EpochETAMeter(epochs)
    # main training loop
    for ep in range(1, epochs+1):
        model.train()
        epoch_meter.start_epoch()

        # train depth
        step_depth = StepETAMeter(len(depth_train_loader))
        running_depth_loss = 0.0
        pbar_d = tqdm(depth_train_loader, desc=f"Epoch {ep}/{epochs} [depth]")
        for x, depth in pbar_d:
            x, depth = x.to(device), depth.to(device)

            depth_coarse, depth_refined = model.forward_depth(x)

            # loss on refined depth
            loss_l1   = F.l1_loss(depth_refined, depth)
            loss_perc = perceptual_loss_vgg(vgg_perc, depth_refined, depth)
            # combined main loss function for model
            loss      = w_l1 * loss_l1 + w_perc * loss_perc
            # zero gradients
            opt.zero_grad()
            # backpropagate
            loss.backward()
            opt.step()
            # keep track of losses on the fly
            running_depth_loss += loss.item() * x.size(0)
            step_depth.update()
            pbar_d.set_postfix(
                loss=loss.item(),
                l1=loss_l1.item(),
                perc=loss_perc.item(),
                eta_s=int(step_depth.eta_seconds))
        # get average loss per sample and avoid division by 0
        avg_depth_loss = running_depth_loss / max(1,len(depth_train))

        # keep track of label segmented training
        step_mask = StepETAMeter(len(mask_train_loader))
        running_mask_ce = 0.0
        # show progress
        pbar_m = tqdm(mask_train_loader, desc=f"Epoch {ep}/{epochs} [mask]")
        for x, label in pbar_m:
            x, label = x.to(device), label.to(device)
            # output probabilities from shared encoder and transformer and label decoder
            logits = model.forward_mask(x)
            loss_seg = ce(logits, label)
            # zero gradients
            opt.zero_grad()
            # backpropagate
            loss_seg.backward()
            opt.step()
            # keep track of losses on the fly
            running_mask_ce += loss_seg.item() * x.size(0)
            step_mask.update()
            pbar_m.set_postfix(ce=loss_seg.item(), eta_s=int(step_mask.eta_seconds))
        # get average loss per sample and avoid division by 0
        avg_mask_ce = running_mask_ce / max(1,len(mask_train_ds))

        # end epoch and eval
        epoch_time = epoch_meter.end_epoch()
        val_depth_l1 = evaluate_depth(model, depth_val_loader, device)
        val_mask_ce, val_mask_acc = evaluate_mask(model, mask_val_loader, device)
        # print out summary stats for epoch
        print(f"Epoch {ep}: time={epoch_time}s | "
            f"train_depth_L1+perc={avg_depth_loss} | train_mask_CE={avg_mask_ce} | "
            f"val_depth_L1={val_depth_l1} | val_mask_CE={val_mask_ce}, val_mask_acc={val_mask_acc} | "
            f"overall_eta≈{int(epoch_meter.overall_eta_seconds)}s")

        # save each epoch as a model checkpoint in case it crashes
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/tri_transformer_depth_logchroma_ep{ep}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print("[CKPT] Saved:", ckpt_path)

    # post training prediction
    print("[VIS] Showing reconstructed RGB, GT depth, and refined predicted depth...")
    visualize_predictions(model, depth_val_loader, mask_val_loader, device)

    # save final model
    os.makedirs("checkpoints", exist_ok=True)
    final_path = "checkpoints/tri_transformer_depth_logchroma_final.pth"
    torch.save(model.state_dict(), final_path)
    print("Saved final:", final_path)

    return model

# main
if __name__ == "__main__":
    train_tri_transformer(
        epochs=500,
        batch_size=8,
        lr=1e-4,
        w_l1=0.8,
        w_perc=0.2,)
