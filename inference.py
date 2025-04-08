from share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch
from cldm.ddim_hacked import DDIMSampler
from annotator.util import resize_image, HWC3
import random
from pytorch_lightning import seed_everything
from einops import rearrange
import numpy as np

# Configs
resume_path = '/work/cvlab/students/bhagavan/SemesterProject/ControlNet/lightning_logs/version_0/checkpoints/epoch=1999-step=450000.ckpt'
batch_size = 100
sd_locked = True
only_mid_control = False
train_or_test = 'test'

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cuda()
model.load_state_dict(load_state_dict(resume_path, location='cuda'))
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

ddim_sampler = DDIMSampler(model)
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=False)
model.eval()

x_samples_list = []
output_image_list = []
for _ in range(1):
    X = next(iter(dataloader))
    output_image = X['jpg']
    control = rearrange(X['hint'].to(torch.float32), 'b h w c -> b c h w').cuda()
    prompt = X['txt']
    with torch.no_grad():
        seed = 42
        seed_everything(seed)
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(prompt)]}
        shape = (4, 16, 16)
        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        # strength = 1.0
        ddim_steps = 50
        num_samples = batch_size
        # model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=True)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (rearrange(x_samples, 'b c h w -> b h w c')).cpu().numpy()
        x_samples_list.append(x_samples)
        output_image_list.append(output_image.cpu().numpy())

x_samples_all = np.concatenate(x_samples_list, axis=0)
output_image_all = np.concatenate(output_image_list, axis=0)

def denormalize_from_minus_one_one(normalized_arr, min_val, max_val):
    # Rearrange min and max to match (b, h, w, c) format
    min_val = rearrange(min_val, '1 c 1 1 -> 1 1 1 c')
    max_val = rearrange(max_val, '1 c 1 1 -> 1 1 1 c')
    return (normalized_arr + 1) * (max_val[:, :, :, :3] - min_val[:, :, :, :3]) / 2 + min_val[:, :, :, :3]

# def denormalize_from_minus_one_one(normalized_arr, min_val, max_val):
#     # Rearrange min and max to match (b, h, w, c) format
#     min_val = rearrange(min_val, '1 c 1 1 -> 1 1 1 c')
#     max_val = rearrange(max_val, '1 c 1 1 -> 1 1 1 c')

#     # Select channel 2 and repeat it three times across the last dimension
#     min_val = min_val[:, :, :, 2:3].repeat(3, axis=-1)
#     max_val = max_val[:, :, :, 2:3].repeat(3, axis=-1)

#     return (normalized_arr + 1) * (max_val - min_val) / 2 + min_val

min_val = np.load("/home/bhagavan/SemesterProject/LDC_NS_2D/128x128/harmonics_lid_driven_cavity_Y_train_min_stats.npy")
max_val = np.load("/home/bhagavan/SemesterProject/LDC_NS_2D/128x128/harmonics_lid_driven_cavity_Y_train_max_stats.npy")

x_samples_all = denormalize_from_minus_one_one(x_samples_all, min_val, max_val)
output_image_all = denormalize_from_minus_one_one(output_image_all, min_val, max_val)

np.save(f"/work/cvlab/students/bhagavan/SemesterProject/ControlNet/lightning_logs/version_0/gt_{train_or_test}.npy", output_image_all)
np.save(f"/work/cvlab/students/bhagavan/SemesterProject/ControlNet/lightning_logs/version_0/preds_{train_or_test}.npy", x_samples_all)