import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from PIL import Image
import numpy as np
from torch import nn
from model import FusionModel2
from dataset import KittiDataset

from torch.utils.data import DataLoader, random_split

Batch_size = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/model2_epoch_20.pth"
DATA_ROOT = "./data"
SEQUENCE = [
    "0000",
    "0001",
    "0002",
    "0003",
    "0004",
    "0005",
    "0006",
    "0007",
    "0008",
    "0009",
    "0010",
    "0011",
    "0012",
    "0013",
    "0014",
    "0015",
    "0016",
    "0017",
    "0018",
    "0019",
    "0020",
]


def visualize_global():
    print(f"LOADING MODEL: {MODEL_PATH} ...")
    model = FusionModel2().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    full_dataset = KittiDataset(root_dir=DATA_ROOT, sequences=SEQUENCE)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    print(f"the test_size is {test_size}")
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )

    Total_test_L = 0
    loss = nn.MSELoss()
    with torch.no_grad():
        for input, target in test_loader:
            input = input.to(DEVICE)
            target = target.to(DEVICE)
            pred = model(input)
            L = loss(pred, target)
            Total_test_L += L.item()
    print(f"The avg loss of the datasets is {Total_test_L / len(test_loader)}")

    for _ in range(10):
        idx = random.randint(0, len(test_dataset) - 1)
        sample_info = test_dataset.samples[idx]
        input_seq_tensor, target_label = test_dataset[idx]
        input_tensor = input_seq_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_output = model(input_tensor)
        pred_ratios = pred_output[0].cpu().numpy()

        last_input = sample_info["inputs"][-1]
        l, t, r, b = last_input["bbox"]

        crop_w = r - l
        crop_h = b - t
        crop_cx = (l + r) / 2
        crop_cy = (t + b) / 2

        add_w = (r - l) * 0.3
        add_h = (b - t) * 0.3
        IMG_W, IMG_H = 1242, 375

        crop_l = max(0, l - add_w)
        crop_t = max(0, t - add_h)
        crop_r = min(r + add_w, IMG_W)
        crop_b = min(b + add_h, IMG_H)

        img_dir = last_input["img_dir"]
        frame_id = last_input["frame_id"]
        full_img_path = os.path.join(img_dir, f"{int(frame_id):06d}.png")
        full_image = Image.open(full_img_path).convert("RGB")

        gt_box = sample_info["target"]
        gt_w = gt_box[2] - gt_box[0]
        gt_h = gt_box[3] - gt_box[1]

        real_crop_w = crop_r - crop_l
        real_crop_h = crop_b - crop_t
        real_crop_cx = (crop_l + crop_r) / 2
        real_crop_cy = (crop_t + crop_b) / 2

        pred_dx = pred_ratios[0] * real_crop_w
        pred_dy = pred_ratios[1] * real_crop_h
        pred_w = pred_ratios[2] * real_crop_w
        pred_h = pred_ratios[3] * real_crop_h

        pred_global_cx = real_crop_cx + pred_dx
        pred_global_cy = real_crop_cy + pred_dy

        pred_final_box = [
            pred_global_cx - pred_w / 2,
            pred_global_cy - pred_h / 2,
            pred_w,
            pred_h,
        ]

        fig, ax = plt.subplots(1, figsize=(12, 4))
        ax.imshow(full_image)
        ax.set_title(f"Global Prediction (Frame {frame_id})")

        rect_gt = patches.Rectangle(
            (gt_box[0], gt_box[1]),
            gt_w,
            gt_h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            label="Ground Truth",
        )

        rect_pr = patches.Rectangle(
            (pred_final_box[0], pred_final_box[1]),
            pred_final_box[2],
            pred_final_box[3],
            linewidth=2,
            edgecolor="#00FF00",
            facecolor="none",
            label="Prediction",
        )

        rect_crop = patches.Rectangle(
            (l, t),
            crop_w,
            crop_h,
            linewidth=1,
            edgecolor="cyan",
            linestyle="--",
            facecolor="none",
            label="Input Crop",
        )

        ax.add_patch(rect_gt)
        ax.add_patch(rect_pr)
        ax.add_patch(rect_crop)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    visualize_global()
