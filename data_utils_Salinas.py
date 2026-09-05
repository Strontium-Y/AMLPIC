import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class HSI_Dataset(Dataset):
    def __init__(self, data_path, gt_path, patch_size=13, transform=None):
        # Load .mat files
        data = sio.loadmat(data_path)
        gt = sio.loadmat(gt_path)

        # ---------- Automatically locate the data array ----------
        data_keys = [k for k in data.keys() if not k.startswith('__')]
        if not data_keys:
            raise ValueError("No array found in data file")
        self.data = data[data_keys[0]].astype(np.float32)
        # Handle data dimensions: expected shape (h, w, bands)
        if self.data.ndim == 4 and self.data.shape[0] == 1:
            self.data = self.data.squeeze(0)
        if self.data.ndim != 3:
            raise ValueError(f"Data dimension error: {self.data.shape}, expected 3D (h, w, bands)")

        # ---------- Label array: directly use key 'salinas_gt' ----------
        # If the key is not 'salinas_gt', modify according to actual situation
        if 'salinas_gt' not in gt:
            # Fallback: automatically find the first 2D array as labels
            gt_keys = [k for k in gt.keys() if not k.startswith('__')]
            self.labels = None
            for k in gt_keys:
                arr = gt[k]
                if arr.ndim == 2:
                    self.labels = arr.astype(np.int64)
                    break
            if self.labels is None:
                raise ValueError("No 2D label array found, please check Salinas_gt.mat file")
        else:
            self.labels = gt['salinas_gt'].astype(np.int64)

        # If labels are 3D with shape (1, h, w) or (h, w, 1), squeeze
        if self.labels.ndim == 3:
            if self.labels.shape[0] == 1:
                self.labels = self.labels.squeeze(0)
            elif self.labels.shape[2] == 1:
                self.labels = self.labels.squeeze(2)

        # Ensure labels match the spatial dimensions of the data
        h_data, w_data = self.data.shape[:2]
        h_label, w_label = self.labels.shape
        if h_data != h_label or w_data != w_label:
            raise ValueError(f"Data shape {self.data.shape[:2]} does not match label shape {self.labels.shape}")

        self.patch_size = patch_size
        self.transform = transform
        self.target_bands = 200   # Number of bands expected by the model

        # Normalize
        self.data = self._normalize(self.data)

        # Extract patches around all non-background pixels
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
        patches = np.array(patches)          # (n_samples, h, w, bands)
        labels = np.array(labels)
        coords = np.array(coords)
        patches = patches.transpose(0, 3, 1, 2)   # (n_samples, bands, h, w)
        return patches, labels, coords

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]          # (bands, h, w)
        label = self.targets[idx] - 1      # Convert to 0-based index

        # Band truncation: Salinas originally has 224 bands, take first 200
        if patch.shape[0] > self.target_bands:
            patch = patch[:self.target_bands, :, :]

        patch = torch.from_numpy(patch).float()
        patch = patch.unsqueeze(0)         # (1, target_bands, h, w)

        if self.transform:
            patch = self.transform(patch)
        return patch, label