from share import *
from torch.utils.data import DataLoader
from dataset import MyDataset
from cldm.model import create_model, load_state_dict
import torch
from cldm.ddim_hacked import DDIMSampler
from pytorch_lightning import seed_everything
from einops import rearrange
import numpy as np

# Configs
project_name = 'GenPhy-with-reconstruction-loss-with-trained-vae'
directory = 'fukkqyk1'
model_name = 'epoch=1499-step=337500.ckpt'
resume_path = f'/work/cvlab/students/bhagavan/SemesterProject/ControlNet/{project_name}/{directory}/checkpoints/{model_name}'
batch_size = 100
sd_locked = True
only_mid_control = False

res = 128
latent_dim = res // 8
dataset_name = 'skelneton'
train_or_test = 'train'

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cuda()
model.load_state_dict(load_state_dict(resume_path, location='cuda'))
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

ddim_sampler = DDIMSampler(model)
dataset = MyDataset(train_or_test, res, dataset_name)
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=False)
model.eval()

x_samples_list = []
output_image_list = []
for X in dataloader:
    output_image = X['jpg']
    control = rearrange(X['hint'].to(torch.float32), 'b h w c -> b c h w').cuda()
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
        output_image_list.append(output_image.cpu().numpy())

x_samples_all = np.concatenate(x_samples_list, axis=0)
output_image_all = np.concatenate(output_image_list, axis=0)

def denormalize_from_minus_one_one(normalized_arr, min_val, max_val):
    # Rearrange min and max to match (b, h, w, c) format
    min_val = rearrange(min_val, '1 c 1 1 -> 1 1 1 c')
    max_val = rearrange(max_val, '1 c 1 1 -> 1 1 1 c')
    return (normalized_arr + 1) * (max_val[:, :, :, :3] - min_val[:, :, :, :3]) / 2 + min_val[:, :, :, :3]

min_val = np.load(f"/work/cvlab/students/bhagavan/SemesterProject/LDC_NS_2D/{res}x{res}/processed/{dataset_name}_lid_driven_cavity_Y_train_min_stats.npy")
max_val = np.load(f"/work/cvlab/students/bhagavan/SemesterProject/LDC_NS_2D/{res}x{res}/processed/{dataset_name}_lid_driven_cavity_Y_train_max_stats.npy")

x_samples_all = denormalize_from_minus_one_one(x_samples_all, min_val, max_val)
output_image_all = denormalize_from_minus_one_one(output_image_all, min_val, max_val)

np.save(f"/work/cvlab/students/bhagavan/SemesterProject/ControlNet/{project_name}/{directory}/{res}_{dataset_name}_gt_{train_or_test}.npy", output_image_all)
np.save(f"/work/cvlab/students/bhagavan/SemesterProject/ControlNet/{project_name}/{directory}/{res}_{dataset_name}_preds_{train_or_test}.npy", x_samples_all)