"""
Chargement du dataset PASCAL VOC (via torchvision) et conversion de ses
annotations (format XML -> dict) vers le format [x_center, y_center, w, h]
normalisé + labels entiers attendu par encode_targets (target_encoder.py).
"""
from torchvision.datasets import VOCDetection

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

CLASS_TO_IDX = {name : idx for idx, name in enumerate(VOC_CLASSES)}

def parse_voc_annotation(annotation, class_to_idx=CLASS_TO_IDX):
    size = annotation["annotation"]["size"]
    img_w, img_h = float(size["width"]), float(size["height"])

    objects = annotation["annotation"]["object"]
    if isinstance(objects, dict):
        objects = [objects]

    boxes, labels = [], []
    for obj in objects:
        bbox = obj["bndbox"]
        xmin, ymin = float(bbox["xmin"]), float(bbox["ymin"])
        xmax, ymax = float(bbox["xmax"]), float(bbox["ymax"])

        x_center = (xmin + xmax) * 0.5 / img_w
        y_center = (ymin + ymax) * 0.5 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        boxes.append([x_center, y_center, w, h])
        labels.append(class_to_idx[obj["name"]])

    return boxes, labels

def load_voc_dataset(root="./data", year="2012", image_set="train", download=True):
    """
    Instancie le dataset PASCAL VOC via torchvision.
    La première exécution avec download=True télécharge les données
    (nécessite une connexion réseau) ; ensuite download=False suffit.
    """
    return VOCDetection(root=root, year=year, image_set=image_set, download=download)
"""
if __name__ == "__main__":
    # --- Test 1 : parse_voc_annotation sur une annotation fabriquée à la
    # main (même structure que ce que renvoie VOCDetection), pour vérifier
    # la logique sans avoir besoin de télécharger le vrai dataset.
    fake_annotation = {
        "annotation": {
            "filename": "exemple.jpg",
            "size": {"width": "500", "height": "400", "depth": "3"},
            "object": [
                {"name": "horse", "bndbox": {"xmin": "50", "ymin": "80", "xmax": "470", "ymax": "380"}},
                {"name": "person", "bndbox": {"xmin": "200", "ymin": "100", "xmax": "300", "ymax": "350"}},
            ],
        }
    }
    boxes, labels = parse_voc_annotation(fake_annotation)
    print("Test avec annotation fabriquée :")
    for box, label in zip(boxes, labels):
        print(f"  classe={VOC_CLASSES[label]} (idx={label}), box(x,y,w,h)={[round(v, 3) for v in box]}")
"""
dataset = load_voc_dataset(download=True)
image, annotation = dataset[0]
boxes, labels = parse_voc_annotation(annotation)
print(image.size, boxes, labels)