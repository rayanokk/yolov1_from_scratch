"""
Convertit une liste de boîtes annotées (vérité terrain) en tenseur cible
au format grille S x S utilisé par YOLO (Redmon et al., 2015).
"""

import torch


def encode_targets(boxes, labels, S=7, B=2, C=20):
    """
    Encode les boîtes vérité-terrain d'une image en tenseur cible YOLO.

    Args:
        boxes: liste de [x_center, y_center, w, h], normalisés entre 0 et 1
               par rapport à l'image entière.
        labels: liste des indices de classe (0 à C-1), même longueur que boxes.
        S: taille de la grille (S x S cellules).
        B: nombre de boîtes prédites par cellule (gardé en argument pour la
           cohérence de l'API, mais la cible elle-même n'en encode qu'une
           seule par cellule -- voir note ci-dessous).
        C: nombre de classes.

    Returns:
        target: tenseur (S, S, C + 5). Pour chaque cellule :
                - les C premières valeurs : one-hot de la classe
                - la valeur suivante (index C) : confiance (1 si objet présent)
                - les 4 dernières : x, y (relatifs à la cellule), w, h
                  (relatifs à l'image entière)

    Note importante : le tenseur de SORTIE du réseau a la forme
    (S, S, B*5 + C) car le modèle propose B boîtes candidates par cellule.
    La CIBLE, elle, n'a besoin que d'une seule boîte par cellule (il n'y a
    qu'une vérité terrain) : c'est le rôle de la fonction de perte (phase 3
    du plan) de décider, à l'entraînement, laquelle des B prédictions est
    "responsable" et doit se rapprocher de cette cible.
    """
    target = torch.zeros((S, S, C + 5))

    for box, label in zip(boxes, labels):
        x, y, w, h = box

        # cellule (i, j) qui contient le centre de la boîte
        # i = ligne (axe y), j = colonne (axe x) -- clamp pour éviter
        # un débordement si x ou y vaut exactement 1.0
        i = min(int(S * y), S - 1)
        j = min(int(S * x), S - 1)

        # coordonnées x, y relatives aux bords de la cellule (dans [0, 1])
        x_cell, y_cell = S * x - j, S * y - i

        if target[i, j, C] == 0:  # cellule pas encore utilisée par un autre objet
            target[i, j, C] = 1  # confiance = présence d'objet
            target[i, j, C + 1:C + 5] = torch.tensor([x_cell, y_cell, w, h])
            target[i, j, label] = 1  # one-hot classe

    return target