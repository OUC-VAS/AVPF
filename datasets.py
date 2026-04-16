import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class AV1M_trainval_dataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split

        self.root_path = self.config["root_path"]
        self.csv_root_path = self.config["csv_root_path"]

        self.df = pd.read_csv(os.path.join(self.csv_root_path, f"{self.split}_labels.csv"))
        self.feats_dir = self.root_path

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feats = np.load(os.path.join(self.feats_dir, row["path"][:-4] + ".npz"), allow_pickle=True)
        label = int(row["label"])

        video = feats['visual']
        audio = feats['audio']

        if self.config.get("apply_l2", False):
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))

        return torch.tensor(video), torch.tensor(audio), label, row["path"][:-4] + ".npz"


class AV1M_test_dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.root_path = Path(self.config["root_path"])
        self.csv_root_path = Path(self.config["csv_root_path"])

        df = pd.read_csv(self.csv_root_path / "test_labels.csv")
        
        lab_map = {}
        for _, row in df.iterrows():
            rel = row["path"]
            if rel.endswith(".mp4"):
                rel = rel[:-4] + ".npz"
            lab_map[str(Path(rel).as_posix())] = int(row["label"])

        self.samples = []
        for npz in self.root_path.rglob("*.npz"):
            rel_from_root = str(npz.relative_to(self.root_path).as_posix())
            label = lab_map.get(rel_from_root)
            if label is not None:
                self.samples.append((npz, label))

        if not self.samples:
            raise RuntimeError(f"No matching .npz under {self.root_path}. Check CSV path normalization.")

        self.apply_l2 = bool(self.config.get("apply_l2", False))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        d = np.load(p, allow_pickle=True)
        
        video = d["visual"]
        audio = d["audio"]
        
        if self.apply_l2:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))
            
        path_str = str(p.relative_to(self.root_path).as_posix())
        return torch.tensor(video), torch.tensor(audio), int(label), path_str


class AVLips_Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.root_path = Path(config["root_path"])
        self.samples = []

        categories = {"0_real": 0, "1_fake": 1}

        for folder_name, label in categories.items():
            folder_path = self.root_path / folder_name
            if folder_path.exists():
                for npz_path in folder_path.rglob("*.npz"):
                    self.samples.append((npz_path, label))
            else:
                print(f"Warning: Folder {folder_path} does not exist.")

        if not self.samples:
            raise RuntimeError(f"No .npz files found in {self.root_path}. Check structure (0_real/1_fake).")

        self.apply_l2 = bool(self.config.get("apply_l2", False))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        d = np.load(p, allow_pickle=True)

        # Handle variations in feature key names across different preprocessing pipelines
        if "visual" in d:
            video = d["visual"]
        elif "video" in d:
            video = d["video"]
        else:
            video = d["visual"] # Fallback

        audio = d["audio"]

        if self.apply_l2:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))

        path_str = str(p.relative_to(self.root_path).as_posix())
        return torch.tensor(video), torch.tensor(audio), int(label), path_str


class FakeAVCeleb_Dataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.root_path = Path(self.config["root_path"])
        self.csv_root_path = Path(self.config["csv_root_path"])

        df = pd.read_csv(self.csv_root_path / f"{split}_split.csv")
        
        lab_map = {}
        for _, row in df.iterrows():
            rel = row["full_path"]
            if rel.endswith(".mp4"):
                rel = rel[:-4] + ".npz"
            lab_map[str(Path(rel).as_posix())] = int(row["category"] != "A")

        self.samples = []
        for npz in self.root_path.rglob("*.npz"):
            rel_from_root = str(npz.relative_to(self.root_path).as_posix())
            label = lab_map.get(rel_from_root)
            if label is not None:
                self.samples.append((npz, label))

        if not self.samples:
            raise RuntimeError(f"No matching .npz under {self.root_path}. Check CSV path normalization.")

        self.apply_l2 = bool(self.config.get("apply_l2", False))

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        d = np.load(p, allow_pickle=True)
        
        video = d["visual"]
        audio = d["audio"]
        
        if self.apply_l2:
            video = video / (np.linalg.norm(video, ord=2, axis=-1, keepdims=True))
            audio = audio / (np.linalg.norm(audio, ord=2, axis=-1, keepdims=True))
            
        path_str = str(p.relative_to(self.root_path).as_posix())
        return torch.tensor(video), torch.tensor(audio), int(label), path_str


def load_data(config, test=False):
    nw = int(config.get("num_workers", 8))
    
    if test:
        if config["name"] == "AV1M":
            test_ds = AV1M_test_dataset(config)
        elif config["name"] == "AVLips":
            test_ds = AVLips_Dataset(config)
        elif config["name"] == "FAVC":
            test_ds = FakeAVCeleb_Dataset(config, split="test")
        else:
            raise ValueError(f"Dataset name error. Expected: AV1M, AVLips, FAVC; Got: {config['name']}")

        test_dl = DataLoader(test_ds, shuffle=False, batch_size=1)
        return test_dl

    else:
        if config["name"] == "AV1M":
            train_ds = AV1M_trainval_dataset(config, split="train")
            val_ds = AV1M_trainval_dataset(config, split="val")
        elif config["name"] == "FAVC":
            train_ds = FakeAVCeleb_Dataset(config, split="train")
            val_ds = FakeAVCeleb_Dataset(config, split="val")
        else:
            raise ValueError(f"Dataset name error. Expected: AV1M, FAVC; Got: {config['name']}")

        train_dl = DataLoader(train_ds, shuffle=True, batch_size=1, num_workers=nw)
        val_dl = DataLoader(val_ds, shuffle=False, batch_size=1, num_workers=nw)
        return train_dl, val_dl
