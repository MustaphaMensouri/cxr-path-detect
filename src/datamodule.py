import pytorch_lightning as pl
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class NIHChestDataset(Dataset):
    def __init__(self, img_dir: str, labels_csv: str, split_files: list, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform

        df = pd.read_csv(labels_csv)
        # Filter to only files that exist in this split's folder
        valid = set(p.name for p in self.img_dir.iterdir())
        self.df = df[df['Image Index'].isin(valid)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row['Image Index']
        image = Image.open(img_path).convert('RGB')  # NIH are grayscale PNGs, convert for timm
        label = int(row['Binary_Label'])
        if self.transform:
            image = self.transform(image)
        return image, label


class CVDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        labels_csv = self.data_dir / 'binary_labels.csv'
        self.train_dataset = NIHChestDataset(
            img_dir=self.data_dir / 'train',
            labels_csv=labels_csv,
            split_files=[],
            transform=self.transform
        )
        self.val_dataset = NIHChestDataset(
            img_dir=self.data_dir / 'val',
            labels_csv=labels_csv,
            split_files=[],
            transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)