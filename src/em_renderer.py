import torch
from torch import nn
from torchtyping import TensorType
import math
import torch.nn.functional as F
import torch.fft as fft


class EMSimulator(nn.Module):
    """
    Given a conformation, this module renders the EM image of the projection on the xy plane.
    """

    def __init__(
        self,
        img_side_len,
        ctf_side_len,
        resolution,
        stds,
        densities,
        accelerating_voltage,
        spherical_aberration,
        amplitude_contrast,
    ):
        super().__init__()

        self.projector_module = Projector(
            side_len=img_side_len,
            resolution=resolution,
            stds=stds,
            densities=densities,
        )

        self.ctf_module = CTF(
            side_len=ctf_side_len,
            resolution=resolution,
            accelerating_voltage=accelerating_voltage,
            spherical_aberration=spherical_aberration,
            amplitude_contrast=amplitude_contrast,
        )

    def forward(
        self,
        conf: TensorType["B", "n_atoms, 3"],
        pose: TensorType["B", 3, 3],
        shift: TensorType["B", 2],
        defocus_u: TensorType["B"],
        defocus_v: TensorType["B"],
        astigm_angle: TensorType["B"],
    ) -> TensorType["B", "side_len", "side_len"]:
        """
        conf - point cloud of the atoms of the molecule conformation

        pose - rotation matrix to apply to the molecule to put it in the orientation in which it was photographed by the electron microscope

        shift - translation in the xy plane to apply to the molecule before projecting it onto the xy plane
        """

        # rotate each molecule to its correct pose before projection
        # shape [B, 1, 3, 3] x [B, n_atoms, 3, 1] -> [B, n_atoms, 3, 1]
        mol = torch.matmul(pose.unsqueeze(1), conf.unsqueeze(-1)).squeeze(-1)

        # drop the z-coordinate and apply the xy translations to the molecules
        mol = mol[:, :, :2] + shift.unsqueeze(1)

        # We chunk into batches containing no more than 180k elements as the projection (and to a lesser extent the CTF) temporarily allocates significant memory during computation. 180k is quite conservative and bigger batches are usually fine. However, we choose conservatively to ensure it runs well on GPU and laptop without memory issues (which can lead to termination).
        max_els_per_batch = 180000
        n_batches = (mol.numel() // max_els_per_batch) + 1

        # chunk all tensors into batches with no more than max_els_per_batch elements
        mol = torch.chunk(mol, n_batches, dim=0)
        defocus_u = torch.chunk(defocus_u, n_batches, dim=0)
        defocus_v = torch.chunk(defocus_v, n_batches, dim=0)
        astigm_angle = torch.chunk(astigm_angle, n_batches, dim=0)

        # the batched output projections will be collected in this list
        batched_outputs_four = []
        batched_outputs_real = []

        # feed the conformations to the projector in a batched fashion
        for i in range(len(mol)):
            # generate noiseless projections

            proj = self.projector_module(mol=mol[i])

            # apply the contrast transfer function to the images
            proj_four, proj_real = self.ctf_module(
                proj=proj,
                defocus_u=defocus_u[i],
                defocus_v=defocus_v[i],
                astigmatism_angle=astigm_angle[i],
            )

            # shape [batch_size, side_len, side_len]
            batched_outputs_four.append(proj_four)
            batched_outputs_real.append(proj_real)

        output_imgs_four = torch.cat(batched_outputs_four, dim=0)
        output_imgs_real = torch.cat(batched_outputs_real, dim=0)
        # shape [B, side_len, side_len]

        return output_imgs_four, output_imgs_real


class Projector(nn.Module):
    def __init__(
        self,
        side_len: int,
        resolution: float,
        stds: TensorType["n_atoms"],
        densities: TensorType["n_atoms"],
    ):
        """
        side_len - the width and height of the output image
        resolution - resolution (or pixel width) of a single pixel, in units of Angstrom
        stds - the standard deviation of the normal distribution pdf of each atom, in units of Angstrom.
        densities - the mass (=2d integral) of the normal distribution pdf of each atom. This is used to more accurately model that atoms with more electrons generate more signal on the EM's detector.
        """
        super().__init__()

        # We want a centered plane, such that the middle pixel is either 0 (odd side_len) or the two middle pixels are equidistant from 0 (even side_len)
        x_max = (side_len - 1) * resolution / 2

        # get the coordinate values for all pixels
        coord = torch.arange(
            start=-x_max,
            # add a small epsilon to ensure end is always included in the range
            end=x_max + 1e-5,
            step=resolution,
        )
        # [side_len]

        # add batch dimension and atom dimension for broadcasting and put the x and y coordinates in different dimensions. Note: in images, the x-axis is the horizontal axis, and the y-axis is the vertical axis. Because arrays are indexed in row-major order, a 2d image array is indexed like array[y][x], which returns the pixel (x,y). It is ESSENTIAL that this is respected, because otherwise all rendered projections are transposed, preventing the network to learn at all. Therefore, the x-coordinates in the line below are in the last dimension, and the y-coordinates are in the penultimate dimension.
        x = coord.reshape((1, 1, 1, side_len))
        y = coord.reshape((1, 1, side_len, 1))

        # compute variance and add broadcasting dimensions for the batch, height and width
        var = stds.square().reshape((1, -1, 1, 1))

        # add broadcasting dimensions for the batch, height and width
        densities = densities.reshape((1, -1, 1, 1))

        # register as buffers all tensors that need to be on the same device as the model's parameters
        self.register_buffer("x", x)
        self.register_buffer("y", y)
        self.register_buffer("var", var)
        self.register_buffer("densities", densities)

    def forward(
        self,
        mol: TensorType["B", "n_atoms", 3],
    ) -> TensorType["B", "side_len", "side_len"]:
        """
        Simulates the (noiseless) projection step of the EM image formation process. Given a point cloud of atoms, this function projects each point on the xy plane and draws a 2d normal distribution pdf around it.

        mol - point clouds representing molecules to be projected. Coordinates in units of Angstrom. The third dimension must at least contain two coordinates; the z-coordinate is optional and will be ignored
        """

        # Get the x and y coordinates of all points. The z coordinate is irrelevant, since we project onto the x-y plane
        x_m = mol[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        y_m = mol[:, :, 1].unsqueeze(-1).unsqueeze(-1)
        # [B, n_atoms, 1, 1] with x and y coordinates, respectively, plus added dimensions for the the height and width of the projection image, which are necessary for broadcasting

        # compute the projections (= Gaussian pdfs) for all atoms by multiplying two 1d Gaussian pdf functions, one for x and one for y. Because x and y are independent, the multivariate normal can be split up by a multiplication of two univariate normals, and this is by far the quickest computation (benchmarked on laptop and GPU, and this implementation is blazingly fast)
        proj = (
            torch.exp(-0.5 / self.var * (self.x - x_m).square())
            * torch.exp(-0.5 / self.var * (self.y - y_m).square())
            / (2 * math.pi * self.var)
        )
        # shape [B, n_atoms, side_len, side_len]

        # multiply the mass of each standard normal with the number of electrons of the atom
        proj = proj * self.densities

        # sum the projections over the atom dimension
        img = proj.sum(dim=1)
        # shape [B, side_len, side_len]

        return img


class CTF(nn.Module):
    def __init__(
        self,
        side_len: int,
        resolution: float,
        accelerating_voltage: float,
        spherical_aberration: float,
        amplitude_contrast: float,
    ):
        """
        side_len - width and height of the ctf in pixels. This can (and often should!) be larger than the width of the cryo-em images. ctf_side_len - img_side_len must be an even number.

        resolution - width of a single pixel in units of Angstrom

        accelerating_voltage - in kV (kilovolt)

        spherical_aberration - spherical aberration coeffient (C_s) in mm (millimeter)

        amplitude_contrast - percentage of image contrast generated by electron amplitude rather than phase.

        Typical values are: resolution=1, accelerating_voltage=300, spherical_aberration=2.7, amplitude_contrast=0.1. Ensure to sanity check that the used values are close to this.

        The default argument values are typical Cryo-em defaults which can be found in cryo-em resources (i.e. papers, Relion and Cryosparc websites, etc). The current specific values are copied from the CryoAI codebase (https://github.com/compSPI/cryoAI).

        The CTF formula and computation is performed exactly as described in equations 3,4,5 of (CTFFIND4, Rohou, 2015, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6760662/)
        """
        super().__init__()

        # compute electron wavelength. Convert input from kV to V.
        self.lamda = self.get_wavelength(accelerating_voltage * 1e3)

        # convert from mm to Angstrom
        self.cs = spherical_aberration * 1e7

        # last term in equation 4 in (CTFFIND4, Rohou, 2015)
        self.phase_contrast_term = math.atan(
            amplitude_contrast / math.sqrt(1 - amplitude_contrast**2)
        )

        # fftfreq returns the spatial frequencies [1/Å] corresponding to (and ordered like) the terms that torch.fft.fft(image) returns when invoked on the image.
        spatial_freqs = fft.fftfreq(side_len, d=resolution)

        # Because our signal (the image) is 2d, we must make a 2d grid of spatial frequencies. x is the horizontal dimension, and y is the vertical dimension. Unsqueeze accordingly.
        x = spatial_freqs.unsqueeze(0)
        y = spatial_freqs.unsqueeze(1)

        # compute the length squared (i.e. radius^2) of each pixel w.r.t. the origin in frequency space.
        r2 = x**2 + y**2
        # shape [side_len, side_len]

        # compute the polar angle [-pi, pi] (= angle w.r.t positive x-axis) of each pixel in frequency space
        polar_angles = torch.atan2(y, x)
        # shape [side_len, side_len]

        # register as buffers all tensors that need to be on the same device as the model's parameters. Also unsqueeze an empty batch dimension for broadcasting.
        self.register_buffer("r2", r2.unsqueeze(0))
        self.register_buffer("polar_angles", polar_angles.unsqueeze(0))

    def get_wavelength(self, V):
        """
        Given the EM's accelerating voltage in units of volt, this function returns the wavelength of electrons in units of Angstrom.

        The exact physical formula is:
        Vr = V + (e/(2*m*c**2))*V**2
        lamda = h / math.sqrt(2*m*e*Vr)

        This is based in the de Broglie wavelength formula and a relativistic correction to the accelerating voltage. See https://www.jeol.com/words/emterms/20121023.071258.php.
        """

        # the wavelength in Angstrom, given the accelerating voltage in volt
        lamda = 12.2639 / math.sqrt(V + 0.97845e-6 * V**2)

        return lamda

    def compute_ctf(
        self,
        defocus_u: TensorType["B"],
        defocus_v: TensorType["B"],
        astigmatism_angle: TensorType["B"],
    ) -> TensorType["B", "ctf_side_len", "ctf_side_len"]:
        """
        Computes the CTF (in frequency space). Should be applied to the image by element-wise multiplying with the image (in frequency space). We compute the CTF exactly as explained in (CTFFIND4, Rohou, 2015, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6760662/).

        defocus_u and defocus_v - the defocus values in units of Angstrom along normal directions, such that defocus_u is the maximum defocus value and defocus_v is the minimum (u and v refer to the axes of this rotated frame). NOTE: defocus_u MUST be greater than or equal to defocus_v. Switching these arguments will yield a faulty CTF computation.

        astigmatism_angle - the angle in radians between the image x-axis and the direction of defocus_u (i.e. the polar angle of the direction of defocus_u in image space)

        Refer to Figure 1 in (CTFFIND4, Rohou, 2015) for further details on these arguments. In an experimental setting, a CTF estimation package such as CTFFIND estimates the two defocus values and astigmatism angle for each micrograph. We assume a constant phase shift of zero (no phase plate).

        Additional resources regarding CTF computation: https://github.com/ml-struct-bio/cryofire/blob/main/src/ctf.py, https://github.com/asarnow/pyem/blob/master/pyem/ctf.py.
        """

        # add dimensions for broadcasting to output shape [B, side_len, side_len]
        defocus_u = defocus_u.reshape((-1, 1, 1))
        defocus_v = defocus_v.reshape((-1, 1, 1))
        astigmatism_angle = astigmatism_angle.reshape((-1, 1, 1))

        # equation 5 in (CTFFIND4, Rohou, 2015). This formula returns the local defocus, a tensor with the defocus value at each pixel.
        local_defocus = 0.5 * (
            defocus_u
            + defocus_v
            + (defocus_u - defocus_v)
            * torch.cos(2 * (self.polar_angles - astigmatism_angle))
        )
        # shape [B, side_len, side_len]

        # equations 3 and 4 in (CTFFIND4, Rohou, 2015)
        ctf = -torch.sin(
            math.pi * self.lamda * self.r2 * local_defocus
            - 0.5 * math.pi * self.cs * self.lamda**3 * self.r2**2
            + self.phase_contrast_term
        )
        # shape [B, side_len, side_len]

        return ctf

    def forward(
        self,
        proj: TensorType["B", "img_side_len", "img_side_len"],
        defocus_u: TensorType["B"],
        defocus_v: TensorType["B"],
        astigmatism_angle: TensorType["B"],
    ) -> TensorType["B", "img_side_len", "img_side_len"]:
        """
        proj - the noiseless(!) 2d projection of the molecule in real image space

        Applies the CTF to the image and returns the result. The input image as well as the output image are in real (i.e. normal) image space.
        """

        # compute the CTF, which lives in frequency space (=Fourier space)
        ctf = self.compute_ctf(defocus_u, defocus_v, astigmatism_angle)

        # zero-pad the projection to the size of the ctf. (ctf_side_len - img_side_len) must be an EVEN number. (this is not strictly necessary but it makes padding easy)
        img_side_len = proj.shape[-1]
        pad = (ctf.shape[-1] - img_side_len) // 2
        proj_padded = F.pad(proj, (pad, pad, pad, pad))

        # apply ctf in Fourier space
        img_four_padded = fft.fftshift(
            fft.fft2(fft.ifftshift(proj_padded, dim=(-1, -2))) * ctf, dim=(-1, -2)
        )

        # inverse transform to real image
        img_real = fft.fftshift(
            fft.ifft2(fft.ifftshift(img_four_padded, dim=(-1, -2))), dim=(-1, -2)
        ).real[:, pad : pad + img_side_len, pad : pad + img_side_len]

        # remove the padding; crop out the image to shape [B, img_side_len, img_side_len]
        img_four = img_four_padded[
            :, pad : pad + img_side_len, pad : pad + img_side_len
        ]

        return img_four, img_real
