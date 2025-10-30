import onnxruntime as ort
import numpy as np
import cv2

#################################
import onnxruntime as ort
from onnxruntime_extensions import get_library_path, onnx_op, PyOp
import numpy as np

# ========================================
# STEP 1: Define all helper functions
# ========================================
def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1, keepdims=True)
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1, keepdims=True)
    return e_x / div

def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])  # Fixed: np.clip instead of .clamp
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]      # Fixed: always use column 0
        py = points[:, 1] + distance[:, i+1]    # Fixed: always use column 1
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def nms(dets, thresh=0.4):  # Fixed: removed 'self', added thresh parameter
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

    keep = nms(pre_det, nms_thresh)  # Fixed: pass nms_thresh
    det = pre_det[keep, :]
    if use_kps:
        kpss = kpss[order, :, :][keep, :, :]
    return det, kpss

# ========================================
# STEP 2: Register the custom operator
# ========================================
@onnx_op(op_type="RetinaFacePostProcess",
         inputs=[PyOp.dt_float] * 9,  # Adjust based on your model
         outputs=[PyOp.dt_float, PyOp.dt_float])
def retinaface_postprocess_op(*raw_outs):
    return retinaface_postprocess(
        list(raw_outs), 
        input_h=640, 
        input_w=640,
        fmc=3, 
        strides=[8, 16, 32], 
        use_kps=True
    )
#########################

# Load your wrapped ONNX model
model_path = "scrfd_with_post_fixed.onnx"

# If using onnxruntime-extensions, register custom ops
try:
    from onnxruntime_extensions import get_library_path
    so = ort.SessionOptions()
    so.register_custom_ops_library(get_library_path())
    print("Custom ops library registered.")
    sess = ort.InferenceSession(model_path, so)
except ImportError:
    print("onnxruntime-extensions not installed, running with plain onnxruntime.")
    sess = ort.InferenceSession(model_path)

# Check input and output names
print("Inputs:", [i.name for i in sess.get_inputs()])
print("Outputs:", [o.name for o in sess.get_outputs()])

# Load a test image (replace with your own image path)
img_path = "0002102.png"  # 🔄 change this to your test face image
orig_img = cv2.imread(img_path)
if orig_img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

# Preprocess (resize to 640x640 for your model)
img = cv2.resize(orig_img, (640, 640))
img_input = img.astype(np.float32)
img_input = img_input.transpose(2, 0, 1)  # HWC → CHW
img_input = np.expand_dims(img_input, axis=0)  # add batch
img_input /= 255.0  # normalize if model expects [0,1]

# Run inference
input_name = sess.get_inputs()[0].name
outputs = sess.run(None, {input_name: img_input})

# Extract detections + keypoints
final_dets, final_kps = outputs
print("Detections shape:", final_dets.shape)  # [N, 5]
print("Keypoints shape:", final_kps.shape)    # [N, 5, 2]

# Draw detections
draw_img = orig_img.copy()
h, w = orig_img.shape[:2]
scale_x, scale_y = w / 640, h / 640

for i, det in enumerate(final_dets):
    x1, y1, x2, y2, score = det
    if score < 0.3:  # confidence threshold
        continue

    # Scale back to original image size
    x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
    cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw keypoints
    kps = final_kps[i]
    for (kx, ky) in kps:
        kx, ky = int(kx * scale_x), int(ky * scale_y)
        cv2.circle(draw_img, (kx, ky), 2, (0, 0, 255), -1)

# Save or display result
cv2.imwrite("detections_result.jpg", draw_img)
print("Detection results saved to detections_result.jpg")

# Uncomment to view in a window (if running locally)
# cv2.imshow("Detections", draw_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
