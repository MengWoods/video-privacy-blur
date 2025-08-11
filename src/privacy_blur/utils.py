import numpy as np

def choose_device(name: str) -> str:
    if name != "auto":
        return {"cpu": "cpu", "cuda": "0", "mps": "mps"}[name]
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def expand_box(x1,y1,x2,y2, scale, W, H):
    w, h = x2-x1, y2-y1
    cx, cy = x1 + w/2, y1 + h/2
    w2, h2 = w*scale, h*scale
    nx1, ny1 = int(max(0, cx - w2/2)), int(max(0, cy - h2/2))
    nx2, ny2 = int(min(W-1, cx + w2/2)), int(min(H-1, cy + h2/2))
    return nx1, ny1, nx2, ny2

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    denom = area_a + area_b - inter + 1e-6
    return inter / denom

def union_box(a,b):
    x1 = min(a[0], b[0]); y1 = min(a[1], b[1])
    x2 = max(a[2], b[2]); y2 = max(a[3], b[3])
    return np.array([x1,y1,x2,y2], dtype=np.float32)

def nms_merge(boxes, iou_thresh=0.5):
    if not boxes: return []
    kept = []
    boxes_arr = np.array(boxes, dtype=np.float32)
    idxs = list(range(len(boxes)))
    while idxs:
        i = idxs.pop(0)
        for j in list(idxs):
            if iou(boxes_arr[i], boxes_arr[j]) > iou_thresh:
                boxes_arr[i] = union_box(boxes_arr[i], boxes_arr[j])
                idxs.remove(j)
        kept.append(tuple(map(int, boxes_arr[i])))
    return kept
