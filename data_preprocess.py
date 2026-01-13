# frame, track_id, type
# truncated? 0~1
# occluded? 0,1,2,3 we need to filter out objects >= 2
# angle
# x1 y1 x2 y2
# 10->16 is for LiDAR, includes 3D info

from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

label_path = "./data/0000.txt"

columns = [
    "frame",
    "track_id",
    "type",
    "truncated",
    "occluded",
    "alpha",
    "bbox_left",
    "bbox_top",
    "bbox_right",
    "bbox_bottom",
]

try:
    df = pd.read_csv(label_path, sep=" ", names=columns, usecols=range(10))
except FileNotFoundError:
    print("can't find the description file .txt")
    exit()

target = df[(df["track_id"] == 0) & (df["type"] == "Van")]
print(f"we found object with ID=0 in a total of {len(target)} frames")

if len(target) > 0:
    sample = target.iloc[10]
    print("\n--- 抽查第 10 个样本的数据 ---")
    print(f"Frame (帧号): {int(sample['frame'])}")
    print(f"Type  (类型): {sample['type']}")
    print(
        f"BBox  (坐标): [{sample['bbox_left']}, {sample['bbox_top']}, {sample['bbox_right']}, {sample['bbox_bottom']}]"
    )

    width = sample["bbox_right"] - sample["bbox_left"]
    height = sample["bbox_bottom"] - sample["bbox_top"]
    print(f"Size  (尺寸): 宽 {width:.1f} x 高 {height:.1f}")
else:
    print("We can't find car with ID = 0")

sample2 = target.iloc[35]
framenum2 = int(sample2["frame"])
imgname2 = f"{framenum2:06d}.png"
I = Image.open(os.path.join("./data/0000", imgname2))
L = sample2["bbox_left"]
T = sample2["bbox_top"]
R = sample2["bbox_right"]
B = sample2["bbox_bottom"]
# crop_box = {
#     max(0, sample2["bbox_left"] - 10),
#     max(0, sample2["bbox_top"] - 10),
#     min(I.width, sample2["bbox_right"] + 10),
#     min(I.height, sample2["bbox_bottom"] + 10),
# }


frame_num = int(sample["frame"])
img_name = f"{frame_num:06d}.png"
img_path = os.path.join("./data/0000", img_name)

if os.path.exists(img_path):
    full_img = Image.open(img_path)

    l = sample["bbox_left"]
    r = sample["bbox_right"]
    t = sample["bbox_top"]
    b = sample["bbox_bottom"]

    pad = 10
    box = (
        max(0, l - pad),
        max(0, t - pad),
        min(full_img.width, r + pad),
        min(full_img.height, b + pad),
    )
    croped = full_img.crop(box)

    final_input = croped.resize((224, 224))

    plt.figure(figsize=(10, 5))

    # 左边画原图 + 红框
    plt.subplot(1, 2, 1)
    plt.title(f"Original Frame {frame_num}")
    plt.imshow(full_img)
    # 在原图上画个红框看看准不准
    rect = plt.Rectangle((l, t), r - l, b - t, fill=False, edgecolor="red", linewidth=2)
    plt.gca().add_patch(rect)

    rect2 = plt.Rectangle(
        (L, T), R - L, B - T, fill=False, edgecolor="blue", linewidth=1
    )
    plt.gca().add_patch(rect2)

    # 右边画我们要喂给 AI 的图
    plt.subplot(1, 2, 2)
    plt.title("Network Input (224x224)")
    plt.imshow(final_input)

    plt.show()  # 这会弹出一个窗口

    print("🎉 可视化成功！右边的图就是将来喂给模型的数据。")
