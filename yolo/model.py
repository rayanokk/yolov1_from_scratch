# ARCHITECTURE_CONFIG décrit la partie convolutive de YOLOv1 (Figure 3 du
# papier). Chaque élément est :
#   - un tuple (kernel_size, out_channels, stride, padding) pour une couche conv
#   - la chaîne "M" pour un maxpool 2x2 stride 2
#   - une liste [tuple_a, tuple_b, N] pour un bloc de deux couches conv répété N fois
ARCHITECTURE_CONFIG = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


import torch.nn as nn
import torch

class CNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        """
        Bloc de base: une convolution 2D suivie d'une activation leaky ReLU
        (pente 0.1). 
        
        in_channels/out_channels définissent le nombre de canaux
        en entrée/sortie de la conv. 

        **kwargs est transmis tel quel à nn.Conv2d (attend typiquement kernel_size,
        stride, padding).
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x: torch.Tensor):
        """
        Applique la convolution puis l'activation à x
        """
        return self.activation(self.conv(x))

class YOLOv1(nn.Module):
    def __init__(self, in_channels: int=3, S: int=7, B: int=2, C: int=20):
        """
        Modèle YOLOv1 complet (backbone convolutif + tête dense)
        
        in_channels = nombre de canaux de l'image (3 pour RGB)

        S, B, C définissent la géométrie de sortie
        """
        super().__init__()
        self.in_channels = in_channels
        self.conv_layers = self._create_conv_layers(ARCHITECTURE_CONFIG)
        self.fc_layers = self._create_fc_layers(S, B, C)

    def _create_conv_layers(self, config: list):
        """
        Construit la partie convolutive en parcourant config.
        Doit suivre le nombre de canaux (in_channels) d'une couche à l'autre pour 
        instancier correctement chaque CNNBlock
        """
        in_channels = self.in_channels
        model = nn.Sequential()
        for i in range(len(config)):
            if type(config[i]) == tuple:
                model.add_module(
                    f"conv_{i}",
                    CNNBlock(
                        in_channels, 
                        config[i][1], 
                        kernel_size=config[i][0], 
                        stride=config[i][2], 
                        padding=config[i][3]
                    )
                )
                in_channels = config[i][1]
            elif type(config[i]) == list:
                for j in range(config[i][-1]):
                    model.add_module(
                        f"conv_{i}_{j}_0",
                        CNNBlock(
                            in_channels,
                            config[i][0][1],
                            kernel_size=config[i][0][0], 
                            stride=config[i][0][2], 
                            padding=config[i][0][3]
                        )
                    )
                    in_channels = config[i][0][1]
                    model.add_module(
                        f"conv_{i}_{j}_1",
                        CNNBlock(
                            in_channels, 
                            config[i][1][1], 
                            kernel_size=config[i][1][0], 
                            stride=config[i][1][2], 
                            padding=config[i][1][3]
                        )
                    )
                    in_channels = config[i][1][1]
            else:
                model.add_module(
                    f"pool_{i}",
                    nn.MaxPool2d(kernel_size=2, stride=2))
        return model

    def _create_fc_layers(self, S: int, B: int, C: int):
        """
        Construit la tête dense finale, qui produit un vecteur plat
        de taille SS(B*5+C) (le reshape en tenseur grill est fait ailleurs, 
        pas dans le modèle)
        """
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (B*5 +C))
        )
        return model

    def forward(self, x: torch.Tensor):
        return self.fc_layers(self.conv_layers(x))

