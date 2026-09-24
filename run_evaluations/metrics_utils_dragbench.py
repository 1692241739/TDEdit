import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lpips
import clip
from PIL import Image
from einops import rearrange

class MetricsCalculator:
    def __init__(self, device):
        self.device = device
        # 严格对应原始代码: AlexNet 版本的 LPIPS
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        # 严格对应原始代码: ViT-B/32 版本的 CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)

    def calculate_drag_similarity(self, src_pil, tgt_pil):
        # 1. LPIPS 预处理: 归一化到 [-1, 1], 尺寸 224x224
        def to_lpips_ts(pil_img):
            img_np = np.array(pil_img).astype(np.float32) / 127.5 - 1.0
            ts = torch.from_numpy(img_np).to(self.device)
            ts = rearrange(ts, "h w c -> 1 c h w")
            return F.interpolate(ts, (224, 224), mode='bilinear')

        src_lpips = to_lpips_ts(src_pil)
        tgt_lpips = to_lpips_ts(tgt_pil)

        with torch.no_grad():
            cur_lpips = self.loss_fn_alex(src_lpips, tgt_lpips).item()

        # 2. CLIP 预处理
        src_clip = self.clip_preprocess(src_pil).unsqueeze(0).to(self.device)
        tgt_clip = self.clip_preprocess(tgt_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            src_feat = self.clip_model.encode_image(src_clip)
            tgt_feat = self.clip_model.encode_image(tgt_clip)
            src_feat /= src_feat.norm(dim=-1, keepdim=True)
            tgt_feat /= tgt_feat.norm(dim=-1, keepdim=True)
            cur_clip_sim = (src_feat * tgt_feat).sum().item()

        return cur_lpips, cur_clip_sim