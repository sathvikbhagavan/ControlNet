import json
import cv2
import numpy as np
import cv2

from torch.utils.data import Dataset
from einops import rearrange


class MyDatasetUnconditional(Dataset):
    def __init__(self, dir_name, train_or_test='test', res=128, dataset_name='harmonics'):
        self.X = np.load(f"/work/cvlab/students/bhagavan/SemesterProject/ControlNet/GenPhy-with-reconstruction-loss-with-trained-vae-stacked/{dir_name}/{res}_{dataset_name}_preds_{train_or_test}_unconditional.npy")
        with open(f"/work/cvlab/students/bhagavan/SemesterProject/ControlNet/GenPhy-with-reconstruction-loss-with-trained-vae-stacked/{dir_name}/{res}_{dataset_name}_reynolds_numbers_{train_or_test}.txt", 'r') as f:
            self.reynolds_numbers = f.readlines()
        self.reynolds_numbers = [float(num.strip()) for num in self.reynolds_numbers]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        reynolds_number =  self.reynolds_numbers[idx]
        prompt = f'{reynolds_number}'
        hint = self.X[idx, :, :, 3:4]  # shape: (H, W, 1)
        # Threshold the hint to binary - less than 1.0 becomes 0, otherwise 1
        # hint = (hint > 1.0).astype(np.uint8) * 255
        # hint = hint.astype(np.uint8)
        # hint = cv2.cvtColor(hint, cv2.COLOR_GRAY2RGB)
        # hint = hint.astype(np.float32) / 255.0  # Normalize
        # Copy three channels from the hint
        hint = np.repeat(hint, 3, axis=2) / 255.0  # shape: (H, W, 3)
        jpg = self.X[idx, :, :, 0:3]  # shape: (H, W, 3)
        return dict(jpg=jpg, txt=prompt, hint=hint)

