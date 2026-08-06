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

    return (inter / (union + eps)).unsqueeze(-1)

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
        calcule la perte YOLO complète entre les prédictions brutes du réseau
        et la cible encodée.
        Args :
            predictions : tenseur (N, S*S*(C+B*5)), sortie brute de YOLOv1.forward
            target : tenseur (N, S, S, C+5), sortie de encode_targets
        
        Returns : 
            Tenseur scalaire : la perte totale du batch, somme de 5 termes
            (localisation, confiance objet, confiance no objet, classification),
            chacun pondéré selon l'équation (3) du papier. 
        """
        S = self.S
        B = self.B
        C = self.C
        mse = self.mse
        lambda_coord = self.lambda_coord
        lambda_noobj = self.lambda_noobj

        predictions = predictions.reshape(-1, S, S, C + B*5)

        conf_preds = predictions[...,C]
        conf_preds = conf_preds.reshape(-1, S, S, B, 5)
        conf_preds = conf_preds[..., 0:1] # (N,S,S,B,1)

        boxes_preds = predictions[..., C:]
        boxes_preds = boxes_preds.reshape(-1, S, S, B, 5)
        boxes_preds = boxes_preds[..., 1:5] # (N,S,S,B,4)

        box_target = target[..., C+1:C+5] # (N,S,S,4)
        ious = []
        for b in range(B):
            box_pred = boxes_preds[...,b, :] # (N,S,S,4)
            iou = IoU(box_pred, box_target)
            ious.append(iou)
        ious = torch.stack(ious)
        best_box = ious.argmax(dim=0) # (N,S,S,1)
        best_iou = ious.max(dim=0).values # (N,S,S,1)

        best_boxes = (
            best_box * boxes_preds[...,1,:]
            + (1 - best_box) * boxes_preds[...,0,:]
        ) # (N,S,S,4)

        best_box_conf = (
            best_box * conf_preds[...,1,:]
            + (1 - best_box) * conf_preds[...,0,:]
        ) # (N,S,S,1)

        obj_mask = target[..., C:C+1] # (N,S,S,1)
        x_pred, y_pred = best_boxes[...,0:1], best_boxes[...,1:2]
        w_pred, h_pred = best_boxes[...,2:3], best_boxes[...,3:4]

        x_target, y_target = box_target[...,0:1], box_target[...,1:2]
        w_target, h_target = box_target[...,2:3], box_target[...,3:4]
        loss1 = lambda_coord * (
            mse(obj_mask * x_pred,
                obj_mask * x_target) +  
            mse(obj_mask * y_pred, 
                obj_mask * y_target)  
        )

        loss2 = lambda_coord * (
            mse(torch.sign(w_pred)*torch.sqrt(torch.abs(w_pred)), 
                torch.sign(w_target)*torch.sqrt(torch.abs(w_target))
            ) + 
            mse(torch.sign(h_pred)*torch.sqrt(torch.abs(h_pred)),
                torch.sign(h_target)*torch.sqrt(torch.abs(h_target))
            )
        )

        loss3 = mse(obj_mask * best_box_conf, obj_mask * best_iou)
        loss4 = lambda_noobj * (
            mse((1-obj_mask) * conf_preds, (1-obj_mask)*best_iou)
        )
        loss5 = mse(obj_mask * predictions[..., :C], 
                    obj_mask * target[..., :C]
                )

        return loss1 + loss2 + loss3 + loss4 + loss5








