import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from PIL import Image
import numpy as np
from torch import nn
from torchvision import transforms
from model import FusionModel2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "/"

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

input_tensor = []

for i in range(10):
    img_dir = os.path.join(root_dir, f"000f{i}.png")
    I = Image.open(img_path).convert("RGB")
    l, t, r, b = l[i], t[i], r[i], b[i]
    add_w = (r - l) * 0.3
    add_h = (b - t) * 0.3

    crop_l = max(0, l - add_w)
    crop_t = max(0, t - add_h)
    crop_r = min(r + add_w, I.width)
    crop_b = min(b + add_h, I.height)

    crop_box = (crop_l, crop_t, crop_r, crop_b)

    crop_I = I.crop(crop_box)
    torch.stack(self.transform(crop_I))

model = FusionModel2().to(DEVICE)
model.load_state_dict(torch.load("/checkpoints/model2_epoch_20.pth", map_location=DEVICE))
model.eval()

with torch.no_grad():
    predict = model(input_tensor)
