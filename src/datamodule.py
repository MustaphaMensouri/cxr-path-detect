import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import lightning as L


LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "No Finding", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]


class XrayDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform):
        self.df = pd.read_csv(csv_path)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.labels = [l for l in LABELS if l in self.df.columns]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.data_dir / row["image_path"]).convert("RGB")
        label = int(self.labels.index(row[self.labels].idxmax()))
        return self.transform(img), label


class XrayDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_tf = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.val_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _loader(self, split, transform, shuffle=False):
        ds = XrayDataset(f"{self.cfg.data_dir}/{split}.csv", self.cfg.data_dir, transform)
        return DataLoader(ds, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, shuffle=shuffle, pin_memory=True)

    def train_dataloader(self): return self._loader("train", self.train_tf, shuffle=True)
    def val_dataloader(self):   return self._loader("val",   self.val_tf)
    def test_dataloader(self):  return self._loader("test",  self.val_tf)