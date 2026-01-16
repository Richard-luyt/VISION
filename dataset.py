import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class KittiDataset(Dataset):
    def __init__(self, root_dir, sequences, seq_len=10):
        self.root_dir = root_dir
        self.seq_len = seq_len

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        cols = [
            "frame",
            "track_id",
            "type",
            "truncated",
            "occluded",
            "alpha",
            "l",
            "t",
            "r",
            "b",
        ]
        self.samples = []

        for seq in sequences:
            txt_path = os.path.join(root_dir, f"{seq}.txt")
            img_dir = os.path.join(root_dir, seq)

            if not os.path.exists(txt_path) or not os.path.exists(img_dir):
                print(f"CAN NOT FIND FILE IN {seq}")
                continue

            try:
                df = pd.read_csv(txt_path, sep=" ", names=cols, usecols=range(10))
            except FileNotFoundError:
                print(f"CANT NOT FIND THE TEXT FILE OF {seq}")
                continue

            df = df[(df["type"] == "Van") | (df["type"] == "Car")]

            for track_id, data in df.groupby("track_id"):
                data = data.sort_values("frame")
                frames = data["frame"].values
                bboxes = data[["l", "t", "r", "b"]].values
                if len(frames) < seq_len + 1:
                    continue

                if "0020" in root_dir:
                    fra = len(self.all_files)
                    split_index = int(fra * 0.8)
                    if self.mode == "train":
                        frames = frames[:split_index]
                        print(f"Highway Mode (Train): {len(frames)} ")

                    elif self.mode == "test":
                        frames = frames[split_index:]
                        print(f"Highway Mode (Test): {len(frames)} ")

                for i in range(len(frames) - seq_len):
                    if (
                        frames[i + seq_len] - frames[i] == seq_len
                    ):  # remove when the car poped in and out
                        input = []
                        for k in range(seq_len):
                            input.append(
                                {
                                    "frame_id": frames[i + k],
                                    "bbox": bboxes[i + k],
                                    "img_dir": img_dir,
                                }
                            )

                        self.samples.append(
                            {"inputs": input, "target": bboxes[i + seq_len]}
                        )
        print("All DATA LOADED")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]

        input_tensor = []

        current_scene_w = None
        current_scene_h = None
        current_scene_cx = None
        current_scene_cy = None

        for i, info in enumerate(item["inputs"]):
            # int(info['frame_id']) is neccessary since info['frame_id'] is a string
            # os.path.join takes in STRINGS
            img_path = os.path.join(info["img_dir"], f"{int(info['frame_id']):06d}.png")
            I = Image.open(img_path).convert("RGB")

            l, t, r, b = info["bbox"]
            add_w = (r - l) * 0.3
            add_h = (b - t) * 0.3

            crop_l = max(0, l - add_w)
            crop_t = max(0, t - add_h)
            crop_r = min(r + add_w, I.width)
            crop_b = min(b + add_h, I.height)

            crop_box = (crop_l, crop_t, crop_r, crop_b)

            crop_I = I.crop(crop_box)
            input_tensor.append(self.transform(crop_I))

            if i == len(item["inputs"]) - 1:
                current_scene_w = crop_r - crop_l
                current_scene_h = crop_b - crop_t
                current_scene_cx = (crop_l + crop_r) / 2
                current_scene_cy = (crop_t + crop_b) / 2

        input_seq = torch.stack(input_tensor)

        # now we calculate the deltas
        # remember to calc delta X and delta H

        # last_input_bbox = item["inputs"][-1]["bbox"]
        # l, t, r, b = last_input_bbox

        # crop_w = r - l
        # crop_h = b - t
        # crop_cx = (l + r) / 2
        # crop_cy = (t + b) / 2

        tgt_box = item["target"]
        tgt_cx = (tgt_box[0] + tgt_box[2]) / 2
        tgt_cy = (tgt_box[1] + tgt_box[3]) / 2
        tgt_w_real = tgt_box[2] - tgt_box[0]
        tgt_h_real = tgt_box[3] - tgt_box[1]

        # the reason for (tgt_cx - current_scene_cx) is:
        # 10 -> 11 is 1.1
        # 1000 -> 1001 is 1.001
        # model will be super sensitive to car's position
        # causing inconsistency
        label = torch.tensor(
            [
                (tgt_cx - current_scene_cx) / current_scene_w,
                (tgt_cy - current_scene_cy) / current_scene_h,
                tgt_w_real / current_scene_w,
                tgt_h_real / current_scene_h,
            ],
            dtype=torch.float32,
        )

        # remember that dataset only returns tensor
        return input_seq, label


if __name__ == "__main__":
    print("1")
    ds = KittiDataset("./data/0000.txt", "./data/0000")
    print(f"Sample 0 Input Shape: {ds[0][0].shape}")
    print(f"sample 0 Label:{ds[0][1]}")
