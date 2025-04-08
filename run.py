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
resume_path = '/work/cvlab/students/bhagavan/SemesterProject/ControlNet/control_sd21_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cuda()
model.load_state_dict(load_state_dict(resume_path, location='cuda'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
trainer = pl.Trainer(accelerator="gpu", devices=1, precision="32-true", max_epochs=1000)

# Train!
# trainer.fit(model, dataloader, ckpt_path=resume_path)
trainer.fit(model, dataloader)
