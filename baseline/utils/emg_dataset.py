from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import torch

# dataset for unsuper
# utils/emg_dataset.py

class UnsupervisedEMGDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted([
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith('.npy')
        ])
        if not self.files:
            raise ValueError(f"未在 {data_dir} 中找到 .npy 文件！")

    def __getitem__(self, idx):
        # 直接加载原始信号，不做任何裁剪或填充
        mixture = np.load(self.files[idx]).astype(np.float32)
        return {
            "mixture": torch.from_numpy(mixture),
            "length": len(mixture),          # 可选：用于调试或日志
            "filename": os.path.basename(self.files[idx])  # 可选
        }

    def __len__(self):
        return len(self.files)


def get_unsupervised_emg_dataloader(data_dir, batch_size=1, num_workers=0):
    """
    注意：由于每个样本长度不同，batch_size 必须为 1，
    否则 DataLoader 无法自动堆叠不同长度的张量。
    """
    dataset = UnsupervisedEMGDataset(data_dir)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=None  # 默认 collate 会在 batch_size>1 时报错（这是好事！）
    )


class EMGMixtureDataset(Dataset):
    # 返回字典
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.npy_files = sorted(list(self.data_dir.glob("*.npy")))
        assert len(self.npy_files) > 0, f"No .npy files found in {data_dir}"

        print(f"[EMGMixtureDataset] Loaded {len(self.npy_files)} samples from {data_dir}")

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        # Load (2, L) array
        data = np.load(self.npy_files[idx]).astype(np.float32)  # shape: (2, L)
        assert data.shape[0] == 2, f"Expected shape (2, L), got {data.shape}"

        clean = data[0]      # shape: (L,)
        noise = data[1]      # shape: (L,)

        return {
                "clean": clean,
                "noise": noise
            }


def get_emg_dataloaders(
    train_dir=None,
    val_dir=None,
    batch_size=16,
    num_workers=4,
    shuffle_train=True,
    shuffle_val=False,
    drop_last_train=True,
    drop_last_val=False
):
    train_loader = None
    val_loader = None

    if train_dir is not None:
        train_dataset = EMGMixtureDataset(train_dir)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            drop_last=drop_last_train,
            pin_memory=True
        )

    if val_dir is not None:
        val_dataset = EMGMixtureDataset(val_dir)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle_val,
            num_workers=num_workers,
            drop_last=drop_last_val,
            pin_memory=True
        )

    return train_loader, val_loader