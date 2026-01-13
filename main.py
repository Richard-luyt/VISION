import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from dataset import KittiDataset
from model import FusionModel2

Batch_size = 8
lr = 0.001
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    print(f"the device is {device}")

    data_root = "./data"

    train_seq = [
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
        "0020",
    ]

    dataset = KittiDataset(root_dir=data_root, sequences=train_seq)
    dataloader = DataLoader(
        dataset,
        batch_size=Batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )
    model = FusionModel2().to(device)
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("____Train start____")

    for epoch in range(epochs):
        T_L = 0
        for i, (inputs, target) in enumerate(dataloader):
            inputs = inputs.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(inputs)
            L = loss(pred, target)
            L.backward()
            optimizer.step()
            T_L += L.item()
            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {L.item():.6f}"
                )

        avg_L = T_L / len(dataloader)
        print(f"The loss of this epoch is {avg_L:.6f}")
        print("-------------------------------------")

        save_path = f"checkpoints/model2_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model Saved: {save_path}")


if __name__ == "__main__":
    train()
