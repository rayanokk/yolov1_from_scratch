import torch
import numpy as np

def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x2 + x1) * 0.5
    cy = (y2 + y1) * 0.5
    w =  (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)

    return torch.stack([cx, cy, w, h], dim=-1)

def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w 
    y2 = cy + 0.5 * h

    return torch.stack([x1, y1, x2, y2], dim=-1)

def box_iou_xyxy(boxes1 : torch.Tensor, boxes2 : torch.Tensor, eps : float=1e-10) -> torch.Tensor:
    x11, y11, x12, y12 = boxes1.unbind(-1)
    x21, y21, x22, y22 = boxes2.unbind(-1)

    inter_x1 = torch.maximum(x11[:, None], x21[None, :])
    inter_x2 = torch.maximum(x12[:, None], x22[None, :])
    inter_y1 = torch.maximum(y11[:, None], y21[None, :])
    inter_y2 = torch.maximum(y12[:, None], y22[None, :])

    inter_w =  (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_boxes1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area_boxes2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)
    union = area_boxes1[:, None] + area_boxes2[None, :] - inter 
    
    return inter/(union+eps)
