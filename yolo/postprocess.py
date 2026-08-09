import torch
def decode_predictions(pred: torch.Tensor, S: int=7, B: int=2, C: int=20):
    """
    Return : 
        liste de longueur N, un élément par image du batch, 
        chacun étant une liste de tuples (class_idx, confidence, x, y, w, h).
    """
    list2 = []
    pred = pred.reshape(-1, S, S, C + 5*B)
    for k in range(pred.shape[0]):
        list1 = []
        for i in range(S):
            for j in range(S):
                for b in range(B):
                    conf = pred[k, i, j, C+5*b]
                    x_cell, y_cell,  w, h = pred[k, i, j, C+5*b+1: C+5*b+5]
                    x = (j + x_cell) / S
                    y = (i + y_cell) / S
                    best_class = torch.argmax(pred[k, i, j, 0:C])
                    class_probability = pred[k, i, j, best_class]
                    score = conf * class_probability
                    list1.append((best_class.item(), score.item(), x.item(), y.item(), w.item(), h.item()))
        list2.append(list1)
    return list2

def nms(boxes: list, iou_threshold: float, prob_threshold: float):
    """
    """
    ...