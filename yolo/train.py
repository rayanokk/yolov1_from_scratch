import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch_directml

from dataset import YOLODataset
from model import YOLOv1
from loss import YoloLoss


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, 
                    optimizer: torch.optim.Optimizer, device: torch.device):
    """
    Retourne la perte moyenne sur l'epoch (somme des pertes de chaque batch / nombre de batches)
    """
    total_loss = 0
    model.train()

    for images, targets in tqdm(loader, leave=False):

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(images)

        loss = loss_fn(predictions, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str):
    """
    Permet de reprendre l'entraînement plus tard sans repartir de zéro.
    """
    checkpoint = {
        'model_state_dict' : model.state_dict(), 
        'optimizer_state_dict' : optimizer.state_dict(),
        'epoch' : epoch
    }

    torch.save(checkpoint, path)

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT YOLOv1")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch_directml.device()
    print(f"[INFO] Device utilisé : {device}")

    print("[INFO] Chargement du dataset...")
    dataset = YOLODataset()
    print(f"[INFO] Dataset chargé : {len(dataset)} images")

    print("[INFO] Création du DataLoader...")
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )
    print(f"[INFO] Nombre de batches par epoch : {len(loader)}")
    print("[INFO] Batch size : 32")

    print("[INFO] Création du modèle YOLOv1...")
    model = YOLOv1()
    model.to(device)
    print("[INFO] Modèle chargé sur le device.")

    print("[INFO] Création de la fonction de perte...")
    loss_fn = YoloLoss()

    print("[INFO] Création de l'optimiseur...")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3, # lr fixé dans un premier temps, contrairement à ce que propose le papier
        momentum=0.9,
        weight_decay=0.0005
    )
    print("[INFO] Optimiseur : SGD")
    print("[INFO] Learning rate : 1e-3")
    print("[INFO] Momentum : 0.9")
    print("[INFO] Weight decay : 0.0005")

    checkpoint = torch.load(
        "checkpoints/checkpoint_epoch_22.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    print(f"Reprise de l'entraînement à l'epoch {start_epoch}")

    print("=" * 60)
    print("🔥 DÉBUT DE L'ENTRAÎNEMENT")
    print("=" * 60)
    for i in tqdm(range(start_epoch, 135)):
        tqdm.write(f"[INFO] Début de l'epoch {i}/135")
        avg_loss = train_one_epoch(model, loader, loss_fn, optimizer, device)
        tqdm.write(f"Epoch {i} terminée — perte moyenne : {avg_loss:.4f}")
        save_checkpoint(model, optimizer, i, path=f"checkpoints/checkpoint_epoch_{i}.pth")
        tqdm.write(f"checkpoint_epoch_{i}")
    print("=" * 60)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 60)

