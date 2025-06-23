import json
import cv2
import numpy as np
import cv2

from torch.utils.data import Dataset
from einops import rearrange


class MyDataset(Dataset):
    def __init__(self, train_or_test='train', res=128, dataset_name='harmonics'):
        self.X = np.load(f"/work/cvlab/students/bhagavan/SemesterProject/LDC_NS_2D/{res}x{res}/processed/{dataset_name}_lid_driven_cavity_X_{train_or_test}.npy")
        self.Y = np.load(f"/work/cvlab/students/bhagavan/SemesterProject/LDC_NS_2D/{res}x{res}/processed/{dataset_name}_lid_driven_cavity_Y_{train_or_test}.npy")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        reynolds_number = self.X[idx, 0, 0, 0]
        prompt = f'{reynolds_number}'
        hint = 255.0 - self.X[idx, 2:3, :, :]  # shape: (1, H, W)
        hint = hint.astype(np.float32) / 255.0  # Normalize
        hint = rearrange(hint, 'c h w -> h w c')  # shape: (H, W, 1)
        y_img = rearrange(self.Y[idx, 0:3, :, :], 'c h w -> h w c')  # shape: (H, W, 3)
        jpg = np.concatenate([y_img, hint], axis=-1)
        return dict(jpg=jpg, txt=prompt)

