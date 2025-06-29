from share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import MyDataset
from cldm.model import create_model
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from ldm.models.autoencoder import AutoencoderKL

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

# Load trained weights
ckpt_path = '/work/cvlab/students/bhagavan/SemesterProject/vae/ControlNet/vae-training/wandb/run-20250603_081629-b263llqe/files/vae_epoch_500.pt'
vae = AutoencoderKL(ddconfig, lossconfig, embed_dim).cuda()
vae.load_state_dict(torch.load(ckpt_path, map_location='cuda'), strict=True)
vae.eval()
wandb_logger = WandbLogger(project="GenPhy-with-reconstruction-loss-with-trained-vae")

# Configs
resume_path = '/work/cvlab/students/bhagavan/SemesterProject/ControlNet/control_sd21_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 5e-5
sd_locked = True
only_mid_control = False
seed_everything(42)

model = create_model('./models/cldm_v21.yaml').cuda()

def load_state_dict_filtered(ckpt_path, location='cuda'):
    state_dict = torch.load(ckpt_path, map_location=location)
    keys_to_remove = [k for k in state_dict if k.startswith('first_stage_model.')]
    for k in keys_to_remove:
        del state_dict[k]
    return state_dict

model.load_state_dict(load_state_dict_filtered(resume_path, location='cuda'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

for param in vae.parameters():
    param.requires_grad = False

model.first_stage_model = vae
vae_requires_grad = any(param.requires_grad for param in model.first_stage_model.parameters())
print(f"VAE parameters require gradients: {vae_requires_grad}")

res = 128
dataset_name = 'harmonics'
dataset = MyDataset('train', res, dataset_name)
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
lr_monitor = LearningRateMonitor(logging_interval='epoch')
trainer = pl.Trainer(accelerator="gpu", devices=1, precision="32-true", max_epochs=1500, logger=wandb_logger, callbacks=[lr_monitor])

# Train!
# trainer.fit(model, dataloader, ckpt_path=resume_path)
trainer.fit(model, dataloader)
