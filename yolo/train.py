import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm 

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = YOLODataset()
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )
    model = YOLOv1()
    model.to(device)
    loss_fn = YoloLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3, # lr fixé dans un premier temps, contrairement à ce que propose le papier
        momentum=0.9,
        weight_decay=0.0005
    )
    for i in tqdm(range(135)):
        avg_loss = train_one_epoch(model, loader, loss_fn, optimizer, device)
        tqdm.write(f"Epoch {i} terminée — perte moyenne : {avg_loss:.4f}")
        if i % 5 == 0:
            save_checkpoint(model, optimizer, i, path=f"checkpoint_epoch_{i}.pth")
            tqdm.write(f"checkpoint_epoch_{i}")

