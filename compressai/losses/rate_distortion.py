import torch
import torch.nn as nn
import math
from compressai.registry import register_criterion
from .functions import *

@register_criterion("RateDistortionLoss")
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    
    def __init__(self, lmbda=200, beta=0.5, return_type="all"):
        super().__init__()
        self.hdr_metric = hdrMetric()
        self.ldr_metric = NLPD_Loss()
        self.beta = beta
        self.lmbda = lmbda
        self.return_type = return_type
        self.hs_fn = torch.nn.L1Loss()
        
    def forward(self, output1, output2, target_hsv, target, smax):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        smin = 5
        
        # Ensure smax is on the correct device and type
        if isinstance(smax, torch.Tensor):
            smax = smax.to(target.device).type_as(target)
        else:
            smax = torch.tensor(smax, device=target.device, dtype=target.dtype)
        
        # Ensure smin is on the correct device and type
        smin = torch.tensor(smin, device=target.device, dtype=target.dtype)
        
        # Calculate BPP loss 1
        out["bpp_loss1"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output1["likelihoods"].values()
        )
        
        # LDR reconstruction loss
        target_hdr = target_hsv.to(target.device).type_as(target)
        ldr_hat = output1["ldr_x_hat"]
        ldr_hat_v = ldr_hat[:, 2, :, :].unsqueeze(1)
        ldr_hat_v = ((ldr_hat_v - ldr_hat_v.min()) / (ldr_hat_v.max() - ldr_hat_v.min()))
        ldr_hat_hs = ldr_hat[:, 0:2, :, :]
        
        target_hdr_v = target_hdr[:, 2, :, :].unsqueeze(1)
        target_hdr_hs = target_hdr[:, 0:2, :, :]
        target_hdr_v2 = ((target_hdr_v - target_hdr_v.min()) / (target_hdr_v.max() - target_hdr_v.min()))
        target_hdr_v2 = (smax - smin) * target_hdr_v2 + smin
        
        out["ldr_loss"] = self.ldr_metric(target_hdr_v2, ldr_hat_v) + 5 * self.hs_fn(ldr_hat_hs, target_hdr_hs)
        distortion1 = out["ldr_loss"]
        
        ###############################################
        # HDR reconstruction loss
        ###############################################
        
        # Calculate BPP loss 2
        out["bpp_loss2"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output2["likelihoods"].values()
        )
        
        target_hdr = target.to(target.device).type_as(target)
        hdr_recon = output2["hdr"]
        
        # Normalize HDR reconstruction
        hdr_recon_max = torch.max(torch.max(torch.max(hdr_recon, 1)[0], 1)[0], 1)[0].unsqueeze(1).unsqueeze(
            1).unsqueeze(1)
        hdr_recon = (hdr_recon) / (hdr_recon_max + 1e-30)
        
        # Normalize target HDR
        target_hdr_max = torch.max(torch.max(torch.max(target_hdr, 1)[0], 1)[0], 1)[0].unsqueeze(1).unsqueeze(
            1).unsqueeze(1)
        target_hdr = (target_hdr) / (target_hdr_max + 1e-30)
        
        out["hdr_loss"] = self.hdr_metric(hdr_recon, target_hdr)
        distortion2 = out["hdr_loss"]
        
        # Combine losses - ensure all components are scalars and on the same device
        lmbda_tensor = torch.tensor(self.lmbda, device=target.device, dtype=target.dtype)
        ten_tensor = torch.tensor(10.0, device=target.device, dtype=target.dtype)
        
        out["loss"] = (lmbda_tensor * distortion1 + 
                      ten_tensor * out["bpp_loss1"] + 
                      lmbda_tensor * distortion2 + 
                      out["bpp_loss2"])
        
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
