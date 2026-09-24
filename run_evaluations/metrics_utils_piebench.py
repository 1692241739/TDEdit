import os

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPScore
from torchmetrics.regression import MeanSquaredError
from torchvision import transforms
from torchvision.transforms import Resize


class VitExtractor:
    BLOCK_KEY = "block"
    QKV_KEY = "qkv"

    def __init__(self, model_name, device):
        # 与 PnPInversion 原版保持一致
        self.model = torch.hub.load("facebookresearch/dino:main", model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers = []
        self.outputs_dict = {VitExtractor.BLOCK_KEY: [], VitExtractor.QKV_KEY: []}

    def _register_hooks(self):
        for block in self.model.blocks:
            self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _reset_outputs(self):
        self.outputs_dict = {VitExtractor.BLOCK_KEY: [], VitExtractor.QKV_KEY: []}

    def _get_block_hook(self):
        def _hook(_model, _inp, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _hook

    def _get_qkv_hook(self):
        def _hook(_model, _inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _hook

    def get_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feat = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._reset_outputs()
        return feat

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feat = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._reset_outputs()
        return feat

    def get_patch_size(self):
        return 8 if "8" in self.model_name else 16

    def get_width_patch_num(self, input_img_shape):
        _b, _c, _h, w = input_img_shape
        return w // self.get_patch_size()

    def get_height_patch_num(self, input_img_shape):
        _b, _c, h, _w = input_img_shape
        return h // self.get_patch_size()

    def get_patch_num(self, input_img_shape):
        return 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))

    def get_head_num(self):
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self):
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    def get_keys_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        # [patch_num, 3, head, dim/head] -> key
        return qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]

    @staticmethod
    def attn_cosine_sim(x, eps=1e-8):
        # 与 PnPInversion 原版一致
        x = x[0]
        norm1 = x.norm(dim=2, keepdim=True)
        factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
        return (x @ x.permute(0, 2, 1)) / factor

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        return self.attn_cosine_sim(concatenated_keys[None, None, ...])


class LossG(torch.nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.extractor = VitExtractor(model_name=cfg["dino_model_name"], device=device)
        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = Resize(cfg["dino_global_patch_size"], max_size=480)
        self.global_transform = transforms.Compose([global_resize_transform, imagenet_norm])

    def calculate_global_ssim_loss(self, outputs, inputs):
        # 与 PnPInversion 原版一致
        loss = 0.0
        for a, b in zip(inputs, outputs):
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss


class MetricsCalculator:
    def __init__(self, device):
        self.device = device
        # 与 PnPInversion 原版配置一致
        clip_model_name = (os.environ.get("TDEDIT_CLIP_MODEL_PATH")
                           or os.environ.get("PIE_CLIP_MODEL_PATH")
                           or "openai/clip-vit-large-patch14")
        try:
            self.clip_metric_calculator = CLIPScore(model_name_or_path=clip_model_name).to(device)
        except Exception:
            if clip_model_name != "openai/clip-vit-large-patch14":
                self.clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
            else:
                raise
        self.psnr_metric_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(device)
        self.mse_metric_calculator = MeanSquaredError().to(device)
        self.ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.structure_distance_metric_calculator = LossG(
            cfg={
                "dino_model_name": "dino_vitb8",
                "dino_global_patch_size": 224,
                "lambda_global_cls": 10.0,
                "lambda_global_ssim": 1.0,
                "lambda_global_identity": 1.0,
                "entire_A_every": 75,
                "lambda_entire_cls": 10,
                "lambda_entire_ssim": 1.0,
            },
            device=device,
        )

    @staticmethod
    def _ensure_mask3(mask_np, image_hw):
        if mask_np is None:
            h, w = image_hw
            return np.zeros((h, w, 3), dtype=np.float32)
        m = np.array(mask_np).astype(np.float32)
        if m.ndim == 2:
            m = m[:, :, None]
        if m.shape[2] == 1:
            m = np.repeat(m, 3, axis=2)
        return m

    def calculate_clip_similarity(self, img, txt, mask=None):
        img = np.array(img)
        if mask is not None:
            img = np.uint8(img * np.array(mask))
        img_tensor = torch.tensor(img).permute(2, 0, 1).to(self.device)
        score = self.clip_metric_calculator(img_tensor, txt)
        return score.cpu().item()

    def calculate_psnr(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255.0
        img_gt = np.array(img_gt).astype(np.float32) / 255.0
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask_pred is not None:
            img_pred = img_pred * np.array(mask_pred).astype(np.float32)
        if mask_gt is not None:
            img_gt = img_gt * np.array(mask_gt).astype(np.float32)
        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return self.psnr_metric_calculator(img_pred_tensor, img_gt_tensor).cpu().item()

    def calculate_lpips(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255.0
        img_gt = np.array(img_gt).astype(np.float32) / 255.0
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask_pred is not None:
            img_pred = img_pred * np.array(mask_pred).astype(np.float32)
        if mask_gt is not None:
            img_gt = img_gt * np.array(mask_gt).astype(np.float32)
        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return self.lpips_metric_calculator(img_pred_tensor * 2 - 1, img_gt_tensor * 2 - 1).cpu().item()

    def calculate_mse(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255.0
        img_gt = np.array(img_gt).astype(np.float32) / 255.0
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask_pred is not None:
            img_pred = img_pred * np.array(mask_pred).astype(np.float32)
        if mask_gt is not None:
            img_gt = img_gt * np.array(mask_gt).astype(np.float32)
        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).to(self.device)
        return self.mse_metric_calculator(img_pred_tensor.contiguous(), img_gt_tensor.contiguous()).cpu().item()

    def calculate_ssim(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255.0
        img_gt = np.array(img_gt).astype(np.float32) / 255.0
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask_pred is not None:
            img_pred = img_pred * np.array(mask_pred).astype(np.float32)
        if mask_gt is not None:
            img_gt = img_gt * np.array(mask_gt).astype(np.float32)
        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return self.ssim_metric_calculator(img_pred_tensor, img_gt_tensor).cpu().item()

    def calculate_structure_distance(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32)
        img_gt = np.array(img_gt).astype(np.float32)
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask_pred is not None:
            img_pred = img_pred * np.array(mask_pred).astype(np.float32)
        if mask_gt is not None:
            img_gt = img_gt * np.array(mask_gt).astype(np.float32)
        img_pred_t = torch.from_numpy(np.transpose(img_pred, axes=(2, 0, 1))).unsqueeze(0).to(self.device)
        img_gt_t = torch.from_numpy(np.transpose(img_gt, axes=(2, 0, 1))).unsqueeze(0).to(self.device)
        structure_distance = self.structure_distance_metric_calculator.calculate_global_ssim_loss(img_gt_t, img_pred_t)
        return float(structure_distance.detach().cpu().numpy())

    def calculate_pie_metrics(self, src_pil, tgt_pil, mask_np, tgt_prompt):
        # 与原版口径保持一致：无额外缩放，背景指标在 unedit 区域计算
        image_h, image_w = np.array(src_pil).shape[:2]
        m_edit = self._ensure_mask3(mask_np, (image_h, image_w))
        m_unedit = 1.0 - m_edit

        result = {
            "Structure Distance": self.calculate_structure_distance(src_pil, tgt_pil, None, None),
            "PSNR": np.nan,
            "LPIPS": np.nan,
            "MSE": np.nan,
            "SSIM": np.nan,
            "CLIP Similarity Whole": self.calculate_clip_similarity(tgt_pil, tgt_prompt, None),
            "CLIP Similarity Edited": np.nan,
        }

        if m_unedit.sum() > 0:
            result["PSNR"] = self.calculate_psnr(src_pil, tgt_pil, m_unedit, m_unedit)
            result["LPIPS"] = self.calculate_lpips(src_pil, tgt_pil, m_unedit, m_unedit)
            result["MSE"] = self.calculate_mse(src_pil, tgt_pil, m_unedit, m_unedit)
            result["SSIM"] = self.calculate_ssim(src_pil, tgt_pil, m_unedit, m_unedit)

        if m_edit.sum() > 0:
            result["CLIP Similarity Edited"] = self.calculate_clip_similarity(tgt_pil, tgt_prompt, m_edit)

        return result
