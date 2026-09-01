import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import time

from model import YOLOv1
from postprocess import nms, decode_predictions, draw_boxes
from data import VOC_CLASSES

def load_model(checkpoint_path: str, device: torch.device, S: int=7, B: int=2, C: int=20):
    """
    Retourne le modèle YOLOv1 chargé sur device
    """
    model = YOLOv1(S=S, B=B, C=C)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model

def preprocess_frame(frame: np.ndarray, img_size: int=224):
    """
    Convertit une frame OpenCV en tenseur d'entrée du modèle
    """
    ...
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    return image

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("checkpoints/checkpoint_epoch_130.pth", device)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        t_debut = time.perf_counter()
        image_tensor = preprocess_frame(frame)
        with torch.no_grad():
            predictions = model(image_tensor)

        decoded = decode_predictions(predictions)
        boxes = nms(
        decoded[0],
        iou_threshold=0.5,
        prob_threshold=0.1
        )

        print("Nombre d'images :", len(decoded))
        print("Nombre de boxes :", len(decoded[0]))
        print("Première box :", decoded[0][0])
   
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annoted = draw_boxes(image_pil, boxes, VOC_CLASSES)
        bgr_annoted = cv2.cvtColor(np.array(annoted), cv2.COLOR_RGB2BGR)

        fps = 1 / (time.perf_counter() - t_debut)
        cv2.putText(
            bgr_annoted, 
            f"fps = {fps}", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0)
            )
        cv2.imshow("YOLOv1", bgr_annoted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
