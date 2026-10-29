import kagglehub
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class VehicleDataset(Dataset):
    def __init__(self, folder, img_size=640):
        self.folder = Path(folder)
        self.img_size = img_size
        self.labels_dir = self.folder / "labels"
        images = sorted((self.folder / "images").glob("*"))
        self.images = [
            p for p in images
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            and (self.labels_dir / f"{p.stem}.txt").exists()
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size))
        img_tensor = torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1) / 255.0

        label_path = self.labels_dir / f"{img_path.stem}.txt"
        with open(label_path, "r", encoding="utf-8") as f:
            boxes = [
                list(map(float, line.strip().split()))
                for line in f if len(line.strip().split()) == 5
            ]

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 5) if boxes else torch.zeros((0, 5), dtype=torch.float32)
        return img_tensor, boxes_tensor

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

def pretretement(data_dir=None, batch_size=8, img_size=640, test_size=0.1, num_workers=0):
    if data_dir is None:
        data_dir = Path(kagglehub.dataset_download("ashfakyeafi/road-vehicle-images-dataset")) / "trafic_data"
    else:
        data_dir = Path(data_dir)

    full_train_ds = VehicleDataset(data_dir / "train", img_size=img_size)
    val_ds = VehicleDataset(data_dir / "valid", img_size=img_size)

    test_len = int(len(full_train_ds) * test_size)
    train_len = len(full_train_ds) - test_len
    train_ds, test_ds = random_split(
        full_train_ds, [train_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, val_loader, test_loader

get_loaders = pretretement

if __name__ == "__main__":
    train_loader, val_loader, test_loader = pretretement(batch_size=4)
    print("Train:", len(train_loader.dataset))
    print("Val:", len(val_loader.dataset))
    print("Test:", len(test_loader.dataset))
