import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class HSI_Dataset(Dataset):
    """Hyperspectral dataset: loads from .mat files and returns (patch, label)"""
    def __init__(self, data_path, gt_path, patch_size=13, transform=None):
        data = sio.loadmat(data_path)
        gt = sio.loadmat(gt_path)
        self.data = data['indian_pines_corrected'].astype(np.float32)
        self.labels = gt['indian_pines_gt'].astype(np.int64)
        self.patch_size = patch_size
        self.transform = transform

        self.data = self._normalize(self.data)
        self.patches, self.targets, self.coords = self._extract_patches()

    def _normalize(self, data):
        h, w, bands = data.shape
        data_reshaped = data.reshape(-1, bands)
        scaler = StandardScaler()
        data_norm = scaler.fit_transform(data_reshaped)
        return data_norm.reshape(h, w, bands)

    def _extract_patches(self):
        h, w, _ = self.data.shape
        pad = self.patch_size // 2
        padded_data = np.pad(self.data, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        padded_labels = np.pad(self.labels, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

        patches = []
        labels = []
        coords = []
        for i in range(h):
            for j in range(w):
                if self.labels[i, j] > 0:
                    patch = padded_data[i:i+self.patch_size, j:j+self.patch_size, :]
                    patches.append(patch)
                    labels.append(self.labels[i, j])
                    coords.append(i)
        patches = np.array(patches)
        labels = np.array(labels)
        coords = np.array(coords)
        patches = patches.transpose(0, 3, 1, 2)        # (n_samples, bands, h, w)
        return patches, labels, coords

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]          # (bands, h, w), bands = 200
        label = self.targets[idx] - 1

        patch = torch.from_numpy(patch).float()
        patch = patch.unsqueeze(0)         # (1, 200, 13, 13)

        if self.transform:
            patch = self.transform(patch)
        return patch, label