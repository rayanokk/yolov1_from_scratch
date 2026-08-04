"""
Dataset PyTorch pour YOLO : combine le chargement/parsing PASCAL VOC
(data.py) et l'encodage de la cible en tenseur grille
(target_encoder.py) pour produire des paires (image, target) prêtes pour 
l'entrainement
"""
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data import load_voc_dataset, parse_voc_annotation, VOC_CLASSES
from target_encoder import encode_targets

class YOLODataset(Dataset):
    """
    Dataset PyTorch renvoyant pour chaque échantillon : 
        image : tenseur (3, img_size, img_size)
        target : tenseur (S, S, C+5)

    L'essentiel du travail est délégué aux fonctions écrites
    dans data.py et target_encoder.py ; cette classe se contente
    de les assembler et de fournir l'interface __len__ / __getitem__ 
    attendue par DataLoader.
    """

    def __init__(self, root = "./data", year="2012", image_set="train",
                 S=7, B=2, C=20, img_size=448, download=True): 
        super().__init__()
        self.voc = load_voc_dataset(root=root, year=year, image_set=image_set, download=download)
        self.S, self.B, self.C = S, B, C
        self.transfrom = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, index):
        image, annotation = self.voc[idx]
        boxes, labels = parse_voc_annotation(annotation)

        image = self.transfrom(image)
        target = encode_targets(boxes, labels, S= self.S, B=self.B, C=self.C)

        return image, target
