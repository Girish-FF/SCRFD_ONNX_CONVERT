import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
import numpy as np

from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint

# -------------------- Utility functions -------------------- #
def distance2bbox(points, distance):
    """Decode distance prediction to bounding box."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return torch.stack([x1, y1, x2, y2], -1)

def distance2kps(points, distance):
    """Decode distance prediction to keypoints."""
    num_kps = distance.shape[1] // 2
    kps = []
    for i in range(num_kps):
        x = points[:, 0] + distance[:, i * 2]
        y = points[:, 1] + distance[:, i * 2 + 1]
        kps.append(x)
        kps.append(y)
    return torch.stack(kps, dim=-1)

# -------------------- Wrapper with Post-Processing -------------------- #
class SCRFDWithPost(nn.Module):
    def __init__(self, scrfd_model, score_thresh=0.5, nms_thresh=0.4, use_kps=True):
        super().__init__()
        self.scrfd = scrfd_model
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.use_kps = use_kps

        # Strides for SCRFD (adjust if using large variants)
        self.strides = [8, 16, 32]

    def forward(self, x):
        outs = self.scrfd(x)
        if isinstance(outs, tuple):
            outs = list(outs)

        num_levels = len(self.strides)
        scores_list, bboxes_list, kpss_list = [], [], []

        _, _, H, W = x.shape
        for i, stride in enumerate(self.strides):
            scores = outs[i].squeeze(0).sigmoid()
            bbox_preds = outs[i + num_levels].squeeze(0) * stride

            if self.use_kps:
                kps_preds = outs[i + 2 * num_levels].squeeze(0) * stride

            feat_h, feat_w = H // stride, W // stride
            yv, xv = torch.meshgrid(
                torch.arange(feat_h, device=x.device),
                torch.arange(feat_w, device=x.device),
                indexing="ij"
            )
            anchor_centers = torch.stack((xv, yv), dim=-1).float()
            anchor_centers = anchor_centers.reshape(-1, 2) * stride

            bboxes = distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores.flatten())
            bboxes_list.append(bboxes)

            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss_list.append(kpss)

        scores = torch.cat(scores_list, dim=0)
        bboxes = torch.cat(bboxes_list, dim=0)
        kpss = torch.cat(kpss_list, dim=0) if self.use_kps else torch.empty((0, 10), device=x.device)

        keep = scores > self.score_thresh
        scores, bboxes, kpss = scores[keep], bboxes[keep], kpss[keep]

        keep_idx = nms(bboxes, scores, self.nms_thresh)
        final_boxes = bboxes[keep_idx]
        final_scores = scores[keep_idx]
        final_kpss = kpss[keep_idx] if self.use_kps else torch.empty((0, 10), device=x.device)

        return final_boxes, final_scores, final_kpss

# -------------------- Conversion Function -------------------- #
def build_scrfd_model(config_path, checkpoint_path, device="cpu"):
    cfg = Config.fromfile(config_path)
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model

def pytorch2onnx(config_path, checkpoint_path, onnx_path, input_shape=(640, 640)):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build base SCRFD
    base_model = build_scrfd_model(config_path, checkpoint_path, device=device)

    # Wrap with post-processing
    model = SCRFDWithPost(base_model, score_thresh=0.5, nms_thresh=0.4, use_kps=True).to(device).eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, *input_shape, device=device)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["bboxes", "scores", "landmarks"],
        opset_version=11,
        dynamic_axes={
            "input": {0: "batch"},
            "bboxes": {0: "num_faces"},
            "scores": {0: "num_faces"},
            "landmarks": {0: "num_faces"}
        }
    )
    print(f"ONNX model exported to {onnx_path}")

# -------------------- Run Example -------------------- #
if __name__ == "__main__":
    config_path = "./configs/scrfd/scrfd_10g_kps.py"  # update with your config
    checkpoint_path = "./scrfd_10g_kps.pth"           # update with your checkpoint
    onnx_path = "./scrfd_with_post.onnx"

    pytorch2onnx(config_path, checkpoint_path, onnx_path, input_shape=(640, 640))
