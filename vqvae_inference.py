import torch
from torch.utils.data import DataLoader
from einops import rearrange
from ldm.models.autoencoder import AutoencoderKL
from dataset import MyDataset
import os
import numpy as np

# --- Config ---

dir = '/work/cvlab/students/bhagavan/SemesterProject/vae/ControlNet/vae-training/wandb/run-20250607_190006-kwsygw18/files'
ckpt_path = f'{dir}/vae_epoch_500.pt'

batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_or_test = 'test'

# --- Dataset ---
dataset = MyDataset(train_or_test)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# --- VAE Model Definition ---
ddconfig = {
    'double_z': True,
    'z_channels': 4,
    'resolution': 128,
    'in_channels': 3,
    'out_ch': 3,
    'ch': 128,
    'ch_mult': [1, 2, 4, 4],
    'num_res_blocks': 2,
    'attn_resolutions': [],
    'dropout': 0.0,
}
lossconfig = {'target': 'torch.nn.Identity'}
embed_dim = ddconfig['z_channels']
vqvae = AutoencoderKL(ddconfig, lossconfig, embed_dim, ckpt_path=ckpt_path)
vqvae = vqvae.cuda()
vqvae.eval()

all_originals = []
all_recons = []
all_latents = []

# --- Inference Loop ---
with torch.no_grad():
    for i, x in enumerate(dataloader):
        print(f'Batch {i+1}/{len(dataloader)}')
        images = rearrange(x['jpg'], 'b h w c -> b c h w').float().to(device)

        # Encode & Decode
        e = vqvae.encode(images)
        # z = posterior.sample()
        z_q, vq_loss = vqvae.quantize(e)
        recon = vqvae.decode(z_q)

        all_originals.append(images.cpu().numpy())
        all_recons.append(recon.cpu().numpy())
        all_latents.append(z_q.cpu().numpy())

np.save(os.path.join(dir, f'originals_{train_or_test}.npy'), np.concatenate(all_originals, axis=0))
np.save(os.path.join(dir, f'reconstructions_{train_or_test}.npy'), np.concatenate(all_recons, axis=0))
np.save(os.path.join(dir, f'latents_{train_or_test}.npy'), np.concatenate(all_latents, axis=0))
