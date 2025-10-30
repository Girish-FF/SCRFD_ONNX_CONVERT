import torch
from torchvision.ops import nms



def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]

    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])

    return torch.stack([x1, y1, x2, y2], dim=-1)


def distance2kps(points, distance, max_shape=None):
    preds = []
    num_kps = distance.shape[1] // 2
    for i in range(num_kps):
        px = points[:, 0] + distance[:, i * 2]
        py = points[:, 1] + distance[:, i * 2 + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return torch.stack(preds, dim=-1)  # (n, num_kps*2)


class SCRFDWithPost(torch.nn.Module):
    def __init__(self, scrfd_model, input_size=(640,640),
                 score_thresh=0.5, nms_thresh=0.4, use_kps=True):
        super().__init__()
        self.scrfd = scrfd_model
        self.input_size = input_size
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.strides = [8, 16, 32]  # or [8,16,32,64,128] for larger variant
        self.use_kps = use_kps

    def forward(self, x):
        # Raw outputs from SCRFD
        outs = self.scrfd(x)  
        num_levels = len(self.strides)

        scores_list, bboxes_list, kpss_list = [], [], []

        _, _, H, W = x.shape
        for i, stride in enumerate(self.strides):
            scores = outs[i].squeeze(0).sigmoid()   # (num_anchors,)
            bbox_preds = outs[i + num_levels].squeeze(0) * stride

            if self.use_kps:
                kps_preds = outs[i + num_levels * 2].squeeze(0) * stride

            # Generate anchor centers
            feat_h, feat_w = H // stride, W // stride
            yv, xv = torch.meshgrid(
                [torch.arange(feat_h), torch.arange(feat_w)],
                indexing="ij"
            )
            anchor_centers = torch.stack((xv, yv), dim=-1).float().reshape(-1, 2).to(x.device)
            anchor_centers = anchor_centers * stride

            # Decode boxes
            bboxes = distance2bbox(anchor_centers, bbox_preds)

            scores_list.append(scores.flatten())
            bboxes_list.append(bboxes)

            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss_list.append(kpss)

        scores = torch.cat(scores_list, dim=0)
        bboxes = torch.cat(bboxes_list, dim=0)
        kpss = torch.cat(kpss_list, dim=0) if self.use_kps else None

        # Apply score threshold
        keep = scores > self.score_thresh
        scores = scores[keep]
        bboxes = bboxes[keep]
        if self.use_kps:
            kpss = kpss[keep]

        # Apply NMS
        keep_idx = nms(bboxes, scores, self.nms_thresh)
        final_boxes = bboxes[keep_idx]
        final_scores = scores[keep_idx]
        final_kpss = kpss[keep_idx] if self.use_kps else None

        return final_boxes, final_scores, final_kpss
