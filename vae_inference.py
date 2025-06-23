import torch
from torch.utils.data import DataLoader
from einops import rearrange
from ldm.models.autoencoder import AutoencoderKL
from dataset import MyDataset
import os
import numpy as np

# --- Config ---

run_name = 'run-20250618_151518-tivmkdgz'
directory = f'/work/cvlab/students/bhagavan/SemesterProject/vae/ControlNet/vae-training-stacked/wandb/{run_name}/files'
ckpt_path = f'{directory}/vae_epoch_500.pt'

batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_or_test = 'test'

# --- Dataset ---
res = 128
dataset_name = 'harmonics'
dataset = MyDataset(train_or_test, res=res, dataset_name=dataset_name)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# --- VAE Model Definition ---
ddconfig = {
    'double_z': True,
    'z_channels': 4,
    'resolution': 128,
    'in_channels': 4,
    'out_ch': 4,
    'ch': 128,
    'ch_mult': [1, 2, 4, 4],
    'num_res_blocks': 2,
    'attn_resolutions': [],
    'dropout': 0.0,
}
lossconfig = {'target': 'torch.nn.Identity'}
embed_dim = ddconfig['z_channels']
vae = AutoencoderKL(ddconfig, lossconfig, embed_dim).to(device)

# --- Load Trained Weights ---
vae.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
vae = vae.cuda()
vae.eval()

all_originals = []
all_recons = []
all_latents = []

# --- Inference Loop ---
with torch.no_grad():
    for i, x in enumerate(dataloader):
        print(f'Batch {i+1}/{len(dataloader)}')
        images = rearrange(x['jpg'], 'b h w c -> b c h w').float().to(device)

        # Encode & Decode
        posterior = vae.encode(images)
        # z = posterior.sample()
        z = posterior.mean
        recon = vae.decode(z)

        all_originals.append(images.cpu().numpy())
        all_recons.append(recon.cpu().numpy())
        all_latents.append(z.cpu().numpy())

np.save(os.path.join(directory, f'originals_{res}_{dataset_name}_{train_or_test}.npy'), np.concatenate(all_originals, axis=0))
np.save(os.path.join(directory, f'reconstructions_{res}_{dataset_name}_{train_or_test}.npy'), np.concatenate(all_recons, axis=0))
np.save(os.path.join(directory, f'latents_{res}_{dataset_name}_{train_or_test}.npy'), np.concatenate(all_latents, axis=0))
