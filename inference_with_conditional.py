from share import *
from torch.utils.data import DataLoader
from dataset_with_uncoditional_pred import MyDatasetUnconditional
from cldm.model import create_model, load_state_dict
import torch
from cldm.ddim_hacked import DDIMSampler
from pytorch_lightning import seed_everything
from einops import rearrange
import numpy as np

# Configs
project_name = 'GenPhy-with-reconstruction-loss-with-trained-vae-stacked'
directory = 'dxh9v0g1'
model_name = 'epoch=1499-step=337500.ckpt'
resume_path = '/work/cvlab/students/bhagavan/SemesterProject/ControlNet/GenPhy-with-reconstruction-loss-with-trained-vae/ta8tgt2x/checkpoints/epoch=1499-step=337500.ckpt'
batch_size = 100
sd_locked = True
only_mid_control = False

res = 128
latent_dim = res // 8
dataset_name = 'harmonics'
train_or_test = 'test'

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cuda()
model.load_state_dict(load_state_dict(resume_path, location='cuda'))
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

ddim_sampler = DDIMSampler(model)
dataset = MyDatasetUnconditional(directory, train_or_test, res, dataset_name)
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=False)
model.eval()

x_samples_list = []
masks_list = []
for X in dataloader:
    control = rearrange(X['hint'].to(torch.float32), 'b h w c -> b c h w').cuda()
    masks_list.append(X['hint'])
    prompt = X['txt']
    with torch.no_grad():
        seed = 42
        seed_everything(seed)
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(prompt)]}
        shape = (4, latent_dim, latent_dim)
        ddim_steps = 50
        num_samples = batch_size
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples, shape, cond, verbose=True)
        x_samples = model.decode_first_stage(samples)
        x_samples = (rearrange(x_samples, 'b c h w -> b h w c')).cpu().numpy()
        x_samples_list.append(x_samples)

x_samples_all = np.concatenate(x_samples_list, axis=0)
masks_list_all = np.concatenate(masks_list, axis=0)

def denormalize_from_minus_one_one(normalized_arr, min_val, max_val):
    # Rearrange min and max to match (b, h, w, c) format
    min_val = rearrange(min_val, '1 c 1 1 -> 1 1 1 c')
    max_val = rearrange(max_val, '1 c 1 1 -> 1 1 1 c')
    return (normalized_arr + 1) * (max_val[:, :, :, :3] - min_val[:, :, :, :3]) / 2 + min_val[:, :, :, :3]

min_val = np.load(f"/work/cvlab/students/bhagavan/SemesterProject/LDC_NS_2D/{res}x{res}/processed/{dataset_name}_lid_driven_cavity_Y_train_min_stats.npy")
max_val = np.load(f"/work/cvlab/students/bhagavan/SemesterProject/LDC_NS_2D/{res}x{res}/processed/{dataset_name}_lid_driven_cavity_Y_train_max_stats.npy")

x_samples_all = denormalize_from_minus_one_one(x_samples_all, min_val, max_val)

np.save(f"/work/cvlab/students/bhagavan/SemesterProject/ControlNet/{project_name}/{directory}/{res}_{dataset_name}_preds_{train_or_test}_conditional.npy", x_samples_all)
np.save(f"/work/cvlab/students/bhagavan/SemesterProject/ControlNet/{project_name}/{directory}/{res}_{dataset_name}_sdf_{train_or_test}_conditional.npy", masks_list_all)