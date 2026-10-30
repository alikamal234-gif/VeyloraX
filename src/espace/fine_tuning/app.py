
from pathlib import Path
import random
import shutil
import yaml
import numpy as np
import torch
import kagglehub
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from ultralytics import YOLO

SEED = 42
IMG_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 10
MODEL_WEIGHTS = "yolov9c.pt"
OUTPUT_DIR = Path("veylorax_dataset").resolve()
RUNS_DIR = Path("runs/veylorax").resolve()


NAMES = [
    "ambulance", "army vehicle", "auto rickshaw", "bicycle", "bus", "car",
    "garbagevan", "human hauler", "minibus", "minivan", "motorbike", "pickup",
    "policecar", "rickshaw", "scooter", "suv", "taxi",
    "three wheelers -CNG-", "truck", "van", "wheelbarrow"
]

# 1. Télécharger le dataset
root = Path(kagglehub.dataset_download("ashfakyeafi/road-vehicle-images-dataset")) / "trafic_data"
images_dir = root / "train" / "images"
labels_dir = root / "train" / "labels"



# 2. Sélectionner les images et labels disponibles
image_paths = sorted(
    p for p in images_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    and (labels_dir / f"{p.stem}.txt").is_file()
)




# 3. Dataset PyTorch conservé pour inspection / usage personnalisé
class VehicleDataset(Dataset):
    def __init__(self, paths, img_size=IMG_SIZE):
        self.paths = paths
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        label_path = labels_dir / f"{image_path.stem}.txt"
        image = Image.open(image_path).convert("RGB").resize((self.img_size, self.img_size))
        image = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        boxes = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) == 5:
                boxes.append([float(v) for v in values])
        targets = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 5)
        return image, targets



# 4. Split reproductible 80/10/10
indices = list(range(len(image_paths)))
random.Random(SEED).shuffle(indices)
n = len(indices)
n_train = int(n * 0.8)
n_val = int(n * 0.1)
train_indices = indices[:n_train]
val_indices = indices[n_train:n_train + n_val]
test_indices = indices[n_train + n_val:]
print(f"Total={n} | Train={len(train_indices)} | Val={len(val_indices)} | Test={len(test_indices)}")



# 5. DataLoaders PyTorch (non utilisés directement par Ultralytics)
dataset = VehicleDataset(image_paths)
def collate_fn(batch):
    return torch.stack([x[0] for x in batch]), [x[1] for x in batch]
train_loader = DataLoader(Subset(dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(Subset(dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(Subset(dataset, test_indices), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)



# 6. Matérialiser le même split sur disque pour Ultralytics
# Attention : ce dossier est régénéré à chaque exécution.
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
for split in ("train", "val", "test"):
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def copy_split(split_indices, split):
    for idx in split_indices:
        src_image = image_paths[idx]
        src_label = labels_dir / f"{src_image.stem}.txt"
        shutil.copy2(src_image, OUTPUT_DIR / "images" / split / src_image.name)
        shutil.copy2(src_label, OUTPUT_DIR / "labels" / split / src_label.name)

copy_split(train_indices, "train")
copy_split(val_indices, "val")
copy_split(test_indices, "test")

# 7. Écrire data.yaml avec chemins absolus corrects
config = {
    "path": str(OUTPUT_DIR),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(NAMES),
    "names": NAMES,
}
yaml_path = OUTPUT_DIR / "data.yaml"
with yaml_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
print("YAML:", yaml_path)



# Important : vérifier que l'ordre NAMES correspond aux IDs des labels originaux.
# 8. Fine-tuning YOLOv9-C pré-entraîné
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_WEIGHTS)


results = model.train(
    data=str(yaml_path), epochs=EPOCHS, imgsz=IMG_SIZE,
    batch=BATCH_SIZE, device=device, project=str(RUNS_DIR), name="finetune", exist_ok=True
)


# 9. Utiliser le meilleur checkpoint plutôt que les poids du dernier epoch
best_path = Path(model.trainer.best)
best_model = YOLO(str(best_path))


# 10. Évaluer Validation et Test (mAP50, mAP50-95)
for split in ("val", "test"):
    metrics = best_model.val(data=str(yaml_path), split=split, imgsz=IMG_SIZE, batch=BATCH_SIZE, device=device)
    print(f"{split.upper()} | mAP@50={metrics.box.map50:.4f} | mAP@50-95={metrics.box.map:.4f}")

# 11. Prédire une image du Test
sample_image = image_paths[test_indices[0]]
predictions = best_model.predict(source=str(sample_image), imgsz=IMG_SIZE, conf=0.25, save=True, device=device)
print("Image test :", sample_image)
print("Détections :", len(predictions[0].boxes))

# 12. Sauvegarder le meilleur modèle fine-tuné en .pt
saved_path = Path("veylorax.pt").resolve()
shutil.copy2(best_path, saved_path)
print("Modèle sauvegardé :", saved_path)
