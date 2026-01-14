import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split

from dataset import KittiDataset
from model import FusionModel2

Batch_size = 8
lr = 0.001
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    print(f"the device is {device}")

    data_root = "./data"

    all_seq = [
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

    full_dataset = KittiDataset(root_dir=data_root, sequences=all_seq)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
    )
    print(
        f"Total: {len(full_dataset)} | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}"
    )

    model = FusionModel2().to(device)
    loss = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    resume_path = "checkpoints/model2_epoch_19.pth"
    # resume_path = None
    S = 0
    if os.path.exists(resume_path):
        state_dict = torch.load(resume_path, map_location=device)
        model.load_state_dict(state_dict)
        S = 19
        print(f"Start From Epoch read from {resume_path}")
    else:
        print("Train from start")

    print("____Train start____")

    for epoch in range(S, epochs):
        T_tain_L = 0
        model.train()
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(inputs)
            L = loss(pred, target)
            L.backward()
            optimizer.step()
            T_tain_L += L.item()
            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {L.item():.6f}"
                )

        avg_train_L = T_tain_L / len(train_loader)

        model.eval()

        T_val_loss = 0

        with torch.no_grad():
            for inputs, target in val_loader:
                inputs = inputs.to(device)
                target = target.to(device)
                pred = model(inputs)
                L = loss(pred, target)
                T_val_loss += L.item()
        avg_val_L = T_val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_train_L:.6f} | "
            f"Val Loss: {avg_val_L:.6f}"
        )

        save_path = f"checkpoints/model2_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model Saved: {save_path}")


if __name__ == "__main__":
    train()
