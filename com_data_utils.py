import torch
import torch.nn.functional as F
import numpy as np
import h5py
import scipy.io as sio
from torch.utils.data import Dataset
from sklearn import preprocessing
import os


class HSI_Dataset(Dataset):
    def __init__(self, mat_path, gt_path, patch_size=9, target_bands=128, use_standardize=True):
        self.patch_size = patch_size
        self.pad = patch_size // 2
        self.target_bands = target_bands

        data = self._load_mat(mat_path)
        labels = self._load_mat(gt_path)

        # Ensure data shape is (H, W, C)
        if data.ndim == 3:
            # If bands are in the first dimension and smaller than height/width, transpose
            if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                data = np.transpose(data, (1, 2, 0))

        # Label processing
        if labels.ndim == 3:
            labels = np.squeeze(labels, axis=0) if labels.shape[0] == 1 else labels
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze(axis=1)

        if use_standardize:
            H, W, C = data.shape
            data_flat = data.reshape(-1, C)
            scaler = preprocessing.StandardScaler()
            data_scaled = scaler.fit_transform(data_flat)
            data = data_scaled.reshape(H, W, C)

        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.pixels = np.argwhere(self.labels > 0)
        self.classes = np.unique(self.labels[self.labels > 0])
        print(f"Loaded {self.__class__.__name__}: {len(self.pixels)} samples, {len(self.classes)} classes, data shape {self.data.shape}")

    def _load_mat(self, filepath):
        # ----- Special handling for Houston dataset -----
        if 'Houston' in filepath or 'houston' in filepath:
            try:
                # Try h5py (v7.3 format)
                if h5py.is_hdf5(filepath):
                    with h5py.File(filepath, 'r') as f:
                        # Common variable names list
                        for key in ['data', 'Houstondata', 'hsi', 'image', 'input']:
                            if key in f:
                                data = f[key][()]
                                break
                        else:
                            # Take first non-system key
                            keys = [k for k in f.keys() if not k.startswith('__')]
                            data = f[keys[0]][()]
                        # If data shape is (bands, h, w), transpose to (h, w, bands)
                        if data.ndim == 3 and data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                            data = np.transpose(data, (1, 2, 0))
                        return data
            except Exception as e:
                print(f"h5py failed for Houston, trying scipy: {e}")
            # Fallback to scipy.io.loadmat
            try:
                mat = sio.loadmat(filepath)
                for key in ['data', 'Houstondata', 'hsi', 'image', 'input']:
                    if key in mat:
                        data = mat[key]
                        break
                else:
                    keys = [k for k in mat.keys() if not k.startswith('__')]
                    data = mat[keys[0]]
                if data.ndim == 3 and data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                    data = np.transpose(data, (1, 2, 0))
                return data
            except Exception as e:
                print(f"Failed to load Houston: {e}")
                raise

        # ----- General loading logic (other datasets) -----
        try:
            # Try h5py (for v7.3 format)
            with h5py.File(filepath, 'r') as f:
                keys = [k for k in f.keys() if not k.startswith('__')]
                data = f[keys[0]][()]
                if data.ndim == 3 and data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                    data = np.transpose(data, (1, 2, 0))
                return data
        except:
            # Fallback to scipy.io.loadmat
            mat = sio.loadmat(filepath)
            keys = [k for k in mat.keys() if not k.startswith('__')]
            data = mat[keys[0]]
            if data.ndim == 3 and data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                data = np.transpose(data, (1, 2, 0))
            return data

    def _fix_bands(self, patch):
        C = patch.shape[0]
        if C < self.target_bands:
            pad = self.target_bands - C
            patch = F.pad(patch, (0, 0, 0, 0, 0, pad))
        elif C > self.target_bands:
            patch = patch[:self.target_bands, :, :]
        return patch

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        h, w = self.pixels[idx]
        label = self.labels[h, w]
        h_start = max(0, h - self.pad)
        h_end = min(self.data.shape[0], h + self.pad + 1)
        w_start = max(0, w - self.pad)
        w_end = min(self.data.shape[1], w + self.pad + 1)
        patch = self.data[h_start:h_end, w_start:w_end, :]
        # Pad if patch size is insufficient
        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
            pad_h_top = self.pad - (h - h_start)
            pad_h_bottom = self.patch_size - patch.shape[0] - pad_h_top
            pad_w_left = self.pad - (w - w_start)
            pad_w_right = self.patch_size - patch.shape[1] - pad_w_left
            patch = np.pad(patch, ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right), (0,0)), mode='constant')
        patch = torch.tensor(patch).permute(2, 0, 1).float()   # (C, H, W)
        patch = self._fix_bands(patch)                         # (target_bands, H, W)
        patch = patch.unsqueeze(0)                             # (1, C, H, W)
        label = torch.tensor(label - 1, dtype=torch.long)
        return patch, label