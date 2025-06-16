import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from dataset import MyDataset
from einops import rearrange
from ldm.models.autoencoder import AutoencoderKL
from lpips import LPIPS
import wandb
import torch.nn.functional as F

# --- Initialize wandb and log config ---
codebook_size = 4096
wandb.init(
    project="vqvae-training",
    dir="./vae-training",
    config={
        "batch_size": 16,
        "learning_rate": 1e-4,
        "optimizer": "Adam",
        "kl_beta": 0.0,
        "vq_weight": 1.0,
        "epochs": 500,
        "save_every": 100,
        "lpips_weight": 0.5,
        "sobel_weight": 0.5,
        "model": "AutoencoderKL",
        "notes": "Training VQVAE with pretrained weights with linear decay LR schedule",
        "z_channels": 4,
        "resolution": 128,
        "ch": 128,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "dropout": 0.0,
        "codebook_size": codebook_size
    }
)
config = wandb.config

# --- Dataset ---
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=4, batch_size=config.batch_size, shuffle=True)

# --- Load pretrained VAE ---
ckpt_path = '/work/cvlab/students/bhagavan/SemesterProject/ControlNet/control_sd21_ini.ckpt'
# ckpt = torch.load(ckpt_path, map_location='cpu')
# vqvae_state_dict = {k.replace('first_stage_model.', ''): v for k, v in ckpt.items() if 'first_stage_model' in k}

# --- VAE Model Init ---
ddconfig = {
    'double_z': True,
    'z_channels': config.z_channels,
    'resolution': config.resolution,
    'in_channels': 3,
    'out_ch': 3,
    'ch': config.ch,
    'ch_mult': config.ch_mult,
    'num_res_blocks': config.num_res_blocks,
    'attn_resolutions': [],
    'dropout': config.dropout,
}
lossconfig = {'target': 'torch.nn.Identity'}
embed_dim = config.z_channels
vqvae = AutoencoderKL(ddconfig, lossconfig, embed_dim, codebook_size=codebook_size, ckpt_path=ckpt_path)
vqvae = vqvae.cuda()

total_params = sum(p.numel() for p in vqvae.parameters())
trainable_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# --- Optimizer & Loss ---
optimizer = torch.optim.Adam(vqvae.parameters(), lr=config.learning_rate)
# def lr_lambda(epoch):
#     if epoch < 100:
#         return 1.0                 # 1.0 × initial_lr = 1e-4
#     elif epoch < 300:
#         return 0.5                 # 0.5 × initial_lr = 5e-5
#     else:
#         return 0.1                 # 0.1 × initial_lr = 1e-5

# scheduler = torch.optim.lr_scheduler.LambdaLR(
#     optimizer,
#     lr_lambda=lr_lambda
# )
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.1,
    total_iters=config.epochs
)
loss_fn = nn.L1Loss()
lpips_loss = LPIPS(net='vgg').cuda()

def sobel_gradient_loss(x, recon):
    sobel_x = torch.tensor([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]],
                            dtype=torch.float32,
                            device=x.device).view(1,1,3,3) / 4.0
    sobel_y = sobel_x.transpose(2, 3)
    C = x.shape[1]
    sobel_x = sobel_x.repeat(C, 1, 1, 1)
    sobel_y = sobel_y.repeat(C, 1, 1, 1)
    grad_x_true = F.conv2d(x,     sobel_x, padding=1, groups=C)
    grad_y_true = F.conv2d(x,     sobel_y, padding=1, groups=C)
    grad_x_rec  = F.conv2d(recon, sobel_x, padding=1, groups=C)
    grad_y_rec  = F.conv2d(recon, sobel_y, padding=1, groups=C)
    loss_x = F.l1_loss(grad_x_rec, grad_x_true, reduction='mean')
    loss_y = F.l1_loss(grad_y_rec, grad_y_true, reduction='mean')
    return loss_x + loss_y

# --- Training ---
for epoch in range(config.epochs):
    vqvae.train()
    epoch_loss = 0
    num_batches = 0

    for x in dataloader:
        images = rearrange(x['jpg'], 'b h w c -> b c h w').float().cuda()
        optimizer.zero_grad()

        e = vqvae.encode(images)            # [B, embed_dim, Hbot, Wbot]

        # —— 3) Quantize → z_q and compute vq_loss ——
        z_q, vq_loss = vqvae.quantize(e)     # z_q: [B, embed_dim, Hbot, Wbot], vq_loss: scalar

        # —— 4) Decode z_q → reconstruction ——
        recon = vqvae.decode(z_q)            # [B, C, H, W]

        # —— 5) Compute reconstruction + perceptual + Sobel losses ——
        recon_loss      = loss_fn(recon, images)               # e.g. L1 
        perceptual_loss = lpips_loss(recon, images).mean()     # LPIPS
        sobel_loss      = sobel_gradient_loss(images, recon)   # Sobel gradient

        # —— 6) Total loss = rec + LPIPS + Sobel + VQ ——
        loss = (
            recon_loss
            + config.lpips_weight * perceptual_loss
            + config.sobel_weight * sobel_loss
            + config.vq_weight   * vq_loss
        )
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        # —— 7) Log batch‐level metrics ——
        wandb.log({
            "step":             num_batches + epoch * len(dataloader),
            "recon_loss":       recon_loss.item(),
            "perceptual_loss":  perceptual_loss.item(),
            "sobel_loss":       sobel_loss.item(),
            "vq_loss":          vq_loss.item(),
            "total_loss":       loss.item(),
            "lr":               scheduler.get_last_lr()[0],
        })

    print(f"Epoch {epoch+1}/{config.epochs} - Avg Loss: {epoch_loss / num_batches:.4f}")
    scheduler.step()
    wandb.log({"epoch_loss": epoch_loss / num_batches, "lr": scheduler.get_last_lr()[0], "epoch": epoch + 1})
    
    # Save checkpoint
    if (epoch + 1) % config.save_every == 0:
        save_path = os.path.join(wandb.run.dir, f"vae_epoch_{epoch+1}.pt")
        torch.save(vqvae.state_dict(), save_path)

print("Training complete.")
wandb.finish()