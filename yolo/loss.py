import torch
import torch.nn as nn 

def IoU(boxes_preds: torch.Tensor, boxes_labels: torch.Tensor = None, eps = 1e-6):
    """
    boxes_preds : [..., 4]
    Calcule l'IoU entre deux jeux de boîtes au format (x,y,w,h). 
    """
    x1, y1, w1, h1 = boxes_preds.unbind(-1)
    x2, y2, w2, h2 = boxes_labels.unbind(-1)

    # Conversion en coordonnées de coins pour calculer l'intersection
    x11 = x1 - w1 * 0.5
    x12 = x1 + w1 * 0.5
    y11 = y1 - h1 * 0.5
    y12 = y1 + h1 * 0.5

    x21 = x2 - w2 * 0.5
    x22 = x2 + w2 * 0.5
    y21 = y2 - h2 * 0.5
    y22 = y2 + h2 * 0.5

    inter_x1 = torch.maximum(x11, x21) # bord gauche
    inter_x2 = torch.minimum(x12, x22) # bord droit
    inter_y1 = torch.maximum(y11, y21) # bord haut
    inter_y2 = torch.minimum(y12, y22) # bord bas

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)

    inter = inter_h * inter_w 
    area_preds = w1 * h1
    area_labels = w2 * h2
    union = area_preds + area_labels - inter 

    return inter / (union + eps)

class YoloLoss(nn.Module):
    def __init__(self, S: int=7, B: int=2, C: int=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions: torch.Tensor, target: torch.Tensor):
        """
        predictions : (N, S*S*(C+B*5))
        target : (N, S, S, C+5)
        """
        S = self.S
        B = self.B
        C = self.C
        predictions = predictions.reshape(-1, S, S, C + B*5)

        boxes_preds = predictions[..., C:]
        boxes_preds= boxes_preds.reshape(-1, S, S, B, 5)
        boxes_preds = boxes_preds[..., 1:5]

        box_target = target[..., C+1:C+5]
        ious = []
        for b in range(B):
            box_pred = boxes_preds[...,b, :]
            iou = IoU(box_pred, box_target)
            ious.append(iou)
        ious = torch.stack(ious, dim=-1)


