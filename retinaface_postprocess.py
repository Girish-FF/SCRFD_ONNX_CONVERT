import numpy as np

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)
def nms(self, dets):
    thresh = self.nms_thresh
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
def retinaface_postprocess(net_outs, input_height, input_width, fmc, strides, use_kps, det_thresh=0.5, nms_thresh=0.4):
    scores_list, bboxes_list, kpss_list = [], [], []

    center_cache = {}

    for idx, stride in enumerate(strides):
        scores = net_outs[idx]
        bbox_preds = net_outs[idx + fmc]

        # squeeze batch dim if needed
        if scores.ndim == 3:
            scores = scores.squeeze(0)
        if bbox_preds.ndim == 3:
            bbox_preds = bbox_preds.squeeze(0)
        bbox_preds = bbox_preds * stride

        if use_kps:
            kps_preds = net_outs[idx + fmc * 2]
            if kps_preds.ndim == 3:
                kps_preds = kps_preds.squeeze(0)
            kps_preds = kps_preds * stride

        height = input_height // stride
        width = input_width // stride
        key = (height, width, stride)

        if key in center_cache:
            anchor_centers = center_cache[key]
        else:
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            center_cache[key] = anchor_centers

        pos_inds = np.where(scores >= det_thresh)[0]
        bboxes = distance2bbox(anchor_centers, bbox_preds)
        pos_scores = scores[pos_inds]
        pos_bboxes = bboxes[pos_inds]
        scores_list.append(pos_scores)
        bboxes_list.append(pos_bboxes)

        if use_kps:
            kpss = distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

    scores = np.vstack(scores_list)
    bboxes = np.vstack(bboxes_list)
    kpss = np.vstack(kpss_list) if use_kps else None

    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    order = scores.ravel().argsort()[::-1]
    pre_det = pre_det[order, :]

    keep = nms(pre_det, nms_thresh)
    det = pre_det[keep, :]
    if use_kps:
        kpss = kpss[order, :, :][keep, :, :]
    return det, kpss


import onnx
from onnx import helper
from onnxruntime_extensions import onnx_op, PyOp

# Dynamically register custom op with N inputs
def register_postprocess_op(num_inputs, input_h, input_w, fmc=3, strides=[8,16,32], use_kps=True):
    @onnx_op(op_type="RetinaFacePostProcess",
             inputs=[PyOp.dt_float] * num_inputs,
             outputs=[PyOp.dt_float, PyOp.dt_float])
    def retinaface_postprocess_op(*raw_outs):
        return retinaface_postprocess(list(raw_outs), input_h, input_w, fmc, strides, use_kps)

# ---- Load model ----
model = onnx.load(r"D:\additionalWork\detectAlignCropRecog_pipeline\models\buffalo_detector.onnx")

# Collect raw output tensor names
raw_outputs = [out.name for out in model.graph.output]
print("Detected raw outputs:", raw_outputs)

# Register op with correct number of inputs
register_postprocess_op(len(raw_outputs), input_h=640, input_w=640)

# ---- Add node that consumes raw outputs ----
node = helper.make_node(
    "RetinaFacePostProcess",
    inputs=raw_outputs,
    outputs=["final_dets", "final_kps"],
    domain="ai.onnx.contrib"
)

# Append node to graph
model.graph.node.append(node)

# (Optional) Adjust graph outputs to only expose final_dets and final_kps
# model.graph.output.clear()
# Remove existing outputs
del model.graph.output[:]

model.graph.output.extend([
    helper.make_tensor_value_info("final_dets", onnx.TensorProto.FLOAT, None),
    helper.make_tensor_value_info("final_kps", onnx.TensorProto.FLOAT, None),
])

# Save wrapped model
onnx.save(model, "scrfd_with_post.onnx")
print("Saved scrfd_with_post.onnx with postprocess node")
