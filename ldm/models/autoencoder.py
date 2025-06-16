import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
from contextlib import contextmanager

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config
from ldm.modules.ema import LitEma


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False
                 ):
        super().__init__()
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.use_ema = ema_decay is not None
        if self.use_ema:
            self.ema_decay = ema_decay
            assert 0. < ema_decay < 1.
            self.model_ema = LitEma(self, decay=ema_decay)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, postfix=""):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+postfix)

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val"+postfix)

        self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        ae_params_list = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(
            self.quant_conv.parameters()) + list(self.post_quant_conv.parameters())
        if self.learn_logvar:
            print(f"{self.__class__.__name__}: Learning logvar")
            ae_params_list.append(self.loss.logvar)
        opt_ae = torch.optim.Adam(ae_params_list,
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, log_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
            if log_ema or self.use_ema:
                with self.ema_scope():
                    xrec_ema, posterior_ema = self(x)
                    if x.shape[1] > 3:
                        # colorize with random projection
                        assert xrec_ema.shape[1] > 3
                        xrec_ema = self.to_rgb(xrec_ema)
                    log["samples_ema"] = self.decode(torch.randn_like(posterior_ema.sample()))
                    log["reconstructions_ema"] = xrec_ema
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

# class AutoencoderKL(pl.LightningModule):
#     def __init__(self,
#                  ddconfig,               # same dict you pass to Encoder/Decoder
#                  lossconfig=None,        # no longer used (you’ll replace with L1/LPIPS/…)
#                  embed_dim=32,           # dimensionality of each codebook vector (D)
#                  codebook_size=1024,      # number of codes K
#                  commitment_cost=0.25,   # β in the VQ‐VAE commit loss
#                  ckpt_path=None,
#                  ignore_keys=[],
#                  image_key="image",
#                  colorize_nlabels=None,
#                  monitor=None,
#                  ema_decay=None,
#                  learn_logvar=False      # not used, but kept for compatibility
#                  ):
#         super().__init__()
#         # We ignore learn_logvar and lossconfig in VQ‐VAE
#         self.image_key      = image_key
#         self.embed_dim      = embed_dim
#         self.codebook_size = codebook_size
#         self.commitment_cost = commitment_cost

#         # ——— Encoder & Decoder (same as before, except we drop double_z) ———
#         # Make a copy of ddconfig but force double_z=False, and z_channels=embed_dim
#         dd = dict(ddconfig)
#         dd["double_z"]   = False
#         dd["z_channels"] = embed_dim

#         self.encoder = Encoder(**dd)   # outputs [B, embed_dim, Hbot, Wbot]
#         self.decoder = Decoder(**dd)   # expects input [B, embed_dim, Hbot, Wbot]

#         # ——— Codebook: K embeddings of dimension D=embed_dim ———
#         self.codebook = nn.Embedding(self.codebook_size, self.embed_dim)
#         nn.init.uniform_(self.codebook.weight, -1.0, 1.0)

#         if ckpt_path is not None:
#             self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

#         if colorize_nlabels is not None:
#             assert isinstance(colorize_nlabels, int)
#             self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

#         self.monitor = monitor

#     def init_from_ckpt(self, path, ignore_keys=list()):
#         """
#         Load matching weights from an old Gaussian‐VAE checkpoint (.pt or .pth),
#         while skipping any layers that no longer exist (quant_conv, post_quant_conv,
#         and codebook), and skipping any size‐mismatched layers.
#         """
#         ckpt = torch.load(path, map_location="cpu")
#         sd_old = {k.replace('first_stage_model.', ''): v for k, v in ckpt.items() if 'first_stage_model' in k}
#         # sd_old = ckpt

#         sd_new = {}
#         own_state = self.state_dict()
#         for k_old, v_old in sd_old.items():
#             # 1) Skip any keys the user specifically asked to ignore:
#             if any(k_old.startswith(ik) for ik in ignore_keys):
#                 continue

#             # 2) Skip any layers that no longer exist in VQ‐VAE:
#             #    - quant_conv, post_quant_conv do not exist anymore
#             if k_old.startswith("quant_conv") or k_old.startswith("post_quant_conv"):
#                 continue

#             # 3) Only copy over if the key also exists in our new model’s state_dict
#             if k_old in own_state:
#                 v_new = own_state[k_old]
#                 # 4) Copy only if shapes match exactly
#                 if v_old.shape == v_new.shape:
#                     sd_new[k_old] = v_old.clone()
#                 else:
#                     # shape mismatch: skip it
#                     print(f"Skipping '{k_old}' (shape {v_old.shape} → {v_new.shape})")
#             else:
#                 # key not found in our new model
#                 # (for example, old VAE might have logged some extra buffers)
#                 continue

#         # Finally load the filtered subset:
#         self.load_state_dict(sd_new, strict=False)
#         print(f"Loaded {len(sd_new)}/{len(own_state)} matching keys from {path}")

#     def encode(self, x):
#         """
#         Encode x → continuous “pre-quantized” embeddings e ∈ ℝ^{B×D×Hbot×Wbot}.
#         """
#         return self.encoder(x)  # [B, embed_dim, Hbot, Wbot]

#     def quantize(self, e):
#         """
#         Given e ∈ [B, D, H, W], flatten to [N, D], pick nearest codebook vector per location,
#         then return:
#         - z_q: quantized latents with a straight-through path back to e (shape [B, D, H, W])
#         - vq_loss = embed_loss + commit_loss
#         """

#         B, D, H, W = e.shape
#         # ——— flatten e to [N, D] ———
#         e_flat = e.permute(0, 2, 3, 1).contiguous().view(-1, D)  # [N, D], where N = B*H*W

#         # ——— compute squared distances to each codebook vector ———
#         embedding = self.codebook.weight  # [K, D]
#         # ||e_flat||^2 → [N,1]
#         e_sq = torch.sum(e_flat**2, dim=1, keepdim=True)        # [N, 1]
#         # ||embedding||^2 → [1,K]
#         emb_sq = torch.sum(embedding**2, dim=1, keepdim=True).T  # [1, K]
#         # dot products → [N, K]
#         dot = torch.matmul(e_flat, embedding.t())               # [N, K]
#         # distances → [N, K]
#         dists = e_sq + emb_sq - 2 * dot                         # [N, K]

#         # ——— get nearest code index for each of the N vectors ———
#         encoding_inds = torch.argmin(dists, dim=1)              # [N]

#         # ——— look up the codebook vectors → [N, D] ———
#         e_q = self.codebook(encoding_inds)                      # [N, D]

#         # ——— reshape back to [B, D, H, W] ———
#         e_q_reshaped = e_q.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

#         # ——— compute VQ losses ———
#         e_q_det = e_q_reshaped.detach()
#         e_det   = e.detach()

#         # push codebook vectors toward encoder output (embedding loss)
#         embed_loss  = torch.mean((e_q_det - e)**2)

#         # push encoder output toward chosen code (commitment loss)
#         commit_loss = self.commitment_cost * torch.mean((e - e_q_det)**2)

#         vq_loss = embed_loss + commit_loss

#         # ——— straight‐through “z_q” for the decoder ———
#         # Forward: z_q == e_q_reshaped
#         # Backward: grads flow into “e”
#         z_q = e + (e_q_reshaped - e).detach()

#         return z_q, vq_loss

#     def decode(self, z_q):
#         """
#         Decode the quantized latent z_q ∈ [B, D, Hbot, Wbot] → reconstructed image.
#         """
#         return self.decoder(z_q)

#     def forward(self, x):
#         """
#         Full VQ-VAE forward:
#           x → Encoder → e → Quantize → z_q → Decoder → x_recon
#         Returns: x_recon, vq_loss
#         """
#         # 1) encode → e
#         e = self.encode(x)                        # [B, D, Hbot, Wbot]
#         # 2) quantize e → z_q, compute vq_loss
#         z_q, _ = self.quantize(e)           # z_q same shape, [B, D, Hbot, Wbot]
#         # 3) decode z_q → x_recon
#         x_recon = self.decode(z_q)                # [B, C, H, W]
#         return x_recon, z_q


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

