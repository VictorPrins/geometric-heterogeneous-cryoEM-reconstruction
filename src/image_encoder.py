from torch import nn
from torchtyping import TensorType


class ImageEncoder(nn.Module):
    """
    We use the image encoder architecture described in the CryoFire paper https://arxiv.org/abs/2210.07387.
    """

    def __init__(self, out_dim, img_side_len):
        super().__init__()

        channels = [1, 64, 64, 256, 256, 256, 256, 256]

        # 5 times 2x2 max pooling devides each side by 32. The cnn output size is equal to all channels and remaining pixels flattened.
        cnn_output_size = channels[-1] * (img_side_len // 32) ** 2

        self.encoder = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(channels[1]),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels[1], channels[2], kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(channels[2]),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels[2], channels[3], kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(channels[3], channels[4], kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(channels[4]),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels[4], channels[5], kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(channels[5], channels[6], kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(channels[6]),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels[6], channels[7], kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(channels[7]),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(cnn_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(
        self, x: TensorType["B", "side_len", "side_len"]
    ) -> TensorType["B", "out_dim"]:
        """
        Input images x must already be z-standardized
        """

        # add a channel index: shape[B, 1, side_len, side_len]
        x = x.unsqueeze(1)

        out = self.encoder(x)

        return out
