from torch import nn
from torchtyping import TensorType


class FullMoleculeDecoder(nn.Module):
    """
    Decoder like they use in the Deepmind paper: decodes all residues in one go, so output dimensionality is equal to out_dim * n_residues. This is reshaped before being returned.
    """

    def __init__(self, n_residues, decoder_net):
        super().__init__()

        self.n_residues = n_residues
        self.net = decoder_net

    def forward(
        self, conf_latent: TensorType["B", "latent_dim"]
    ) -> TensorType["B", "n_residues", "out_dim"]:
        x = self.net(conf_latent)
        # shape [B, n_residues * 3]

        out = x.reshape((-1, self.n_residues, 3))

        return out
