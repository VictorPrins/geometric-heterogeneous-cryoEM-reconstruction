import torch
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from em_renderer import EMSimulator
from molecule_model import CAlphaModel
from network_logging import *
from ml_utils import MLP
from image_encoder import ImageEncoder
from conf_decoder import FullMoleculeDecoder
from evaluation import compute_val_metrics


class Hetero_Cryo_Net(LightningModule):
    """
    Heterogeneous protein reconstruction network.
    """

    def __init__(
        self,
        latent_dim,
        dataset,
        ctf_side_len,
        output_norm_shift,
        output_norm_scale,
        dcc_loss_factor,
        geom_loss_factor,
        lr,
    ):
        super().__init__()

        # ignore dataset, otherwise the whole dataset is saved with every checkpoint
        self.save_hyperparameters(ignore=["dataset"])

        self.dataset = dataset
        self.dcc_factor = dcc_loss_factor
        self.geom_factor = geom_loss_factor
        self.lr = lr
        self.out_norm_shift = output_norm_shift
        self.out_norm_scale = output_norm_scale
        self.ctf_side_len = ctf_side_len

        n_residues = len(dataset.res_names)

        self.molecule_model = CAlphaModel(
            prior_conf=dataset.prior_conf,
            res_names=dataset.res_names,
            n_res_total=n_residues,
        )

        self.image_to_latent = ImageEncoder(
            out_dim=latent_dim,
            img_side_len=dataset.side_len,
        )

        self.conf_decoder = FullMoleculeDecoder(
            n_residues=n_residues,
            decoder_net=MLP(
                in_dim=latent_dim,
                out_dim=3 * n_residues,
                n_layers=7,
                use_layer_norm=False,
            ),
        )

        self.em_simulator = EMSimulator(
            img_side_len=dataset.side_len,
            ctf_side_len=ctf_side_len,
            resolution=dataset.resolution,
            stds=self.molecule_model.stds,
            densities=self.molecule_model.densities,
            accelerating_voltage=dataset.accelerating_voltage,
            spherical_aberration=dataset.spherical_aberration,
            amplitude_contrast=dataset.amplitude_contrast,
        )

        # init bias of second node on 1
        self.output_norm_estimator = MLP(in_dim=1, out_dim=2, n_layers=3)
        # initialize the bias of the second node at 1, because the rescaling should start around 1 (meaning no output rescaling)
        with torch.no_grad():
            self.output_norm_estimator.mlp[-1].bias[1] = 1

        # these computations are needed for the geometry degradation loss
        # get positions of C-alpha atoms [n_residues, 3]
        prior_conf = self.molecule_model.prior

        # Get all combinations (i,j) of all residues, excluding pairs with itself (i.e. not (i,i)) and excluding permutations (i.e. (i,j) iff (j,i) is not already in there). These are all the pairs in the upper triangular (excluding diagonal) of the Euclidian distance matrix. The returned tensor has shape [n_residues * (n_residues - 1) / 2, 2]
        i_j_pairs = torch.combinations(torch.arange(n_residues))

        # register as buffer so they are on the correct device
        self.register_buffer("i_idx", i_j_pairs[:, 0])
        self.register_buffer("j_idx", i_j_pairs[:, 1])

        # compute the (flattened) upper triangular (excluding diagonal) of the Euclidian distance matrix (squared and logged)
        prior_edm_sq_log = prior_conf[self.i_idx] - prior_conf[self.j_idx]
        prior_edm_sq_log = prior_edm_sq_log.square().sum(dim=-1).log()

        # register as buffer to ensure the tensor is on the correct device
        # shape [n_residues * (n_residues - 1) / 2]
        self.register_buffer("prior_edm_sq_log", prior_edm_sq_log)

        # list to store the outputs of the validation loop
        self.val_step_outputs = []

    def forward(self, batch):
        # batch has keys "img", "idx", "defocus_u", "defocus_v", "astigm_angle"
        # the input images must already be z-standardized
        # We perform no initial shift or rotation

        latent = self.image_to_latent(batch["img"])

        # compute deform tensor of shape [B, n_residues, 3]
        deformation_tensor = self.conf_decoder(latent)

        # apply the deformations to the prior to get the predicted conformations
        pred_conf = self.molecule_model(deformation_tensor)
        pose = batch["pose"]

        # the particles are already centered in the input images
        no_shift = torch.zeros((1, 2), device=self.device)

        img_four, img_real = self.em_simulator(
            conf=pred_conf,
            pose=pose,
            shift=no_shift,
            defocus_u=batch["defocus_u"],
            defocus_v=batch["defocus_v"],
            astigm_angle=batch["astigm_angle"],
        )

        dummy_input = torch.ones((1, 1), device=self.device)
        out_norm = self.output_norm_estimator(dummy_input).squeeze(0)

        # shift and scale the output image by the fixed values, and then by the learnt values
        img_real = (img_real - self.out_norm_shift) / self.out_norm_scale
        img_real = (img_real - out_norm[0]) / out_norm[1].abs()

        pred = {
            "img_real": img_real,  # shape [B, side_len, side_len]
            "conf": pred_conf,  # shape [B, n_residues, 3]
            "transl": deformation_tensor,  # shape [B, n_residues, 3]
            "out_norm": out_norm,  # shape [2]
        }

        return pred

    def compute_geom_deg_loss(self, pred):
        """
        Compute the geometry degradation loss.
        """

        pred_conf = pred["conf"]
        # [B, n_residues, 3]

        pred_edm_sq_log = pred_conf[:, self.i_idx] - pred_conf[:, self.j_idx]
        pred_edm_sq_log = pred_edm_sq_log.square().sum(dim=-1).log()
        # [B, n_residues * (n_residues - 1) / 2]

        # unsqueeze a batch dimension to the prior edm, and compute the loss
        loss = (pred_edm_sq_log - self.prior_edm_sq_log.unsqueeze(0)).square().mean()

        return loss

    def compute_dcc_loss(self, pred):
        """
        Computes the DCC (Distance between Consecutive C-alpha atoms) loss, defined as the MSE between a) the distance between a pair of C-alpha atoms in neighbouring residues, and b) 3.8 Angstrom.

        According to (Chakraborty, Venkatramani, et al, 2013, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3892923/), neighbouring C-alpha atoms have a distance very close to 3.8 Angstrom in essentially all proteins.

        This loss can also be viewed as a variation on the 'Backbone continuity loss' in (Rosenbaum et al, 2021).
        """

        target_dcc = 3.8

        # compute the C-alpha-C-alpha distances in the predicted conformation
        dccs = self.molecule_model.dcc(pred["conf"])
        # list containing tensors of shape [B, n_residues - 1] with the DDCs

        # compute DCC loss
        loss = (dccs - target_dcc).square().mean()

        return loss

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)

        # compute centroids of predicted conformationes - shape [B, 3]
        centroid = torch.mean(pred["conf"], dim=1)
        # compute the mean squared deviation of the centroids from 0
        centering_loss = centroid.square().mean()

        dcc_loss = self.compute_dcc_loss(pred)
        geometry_deg_loss = self.compute_geom_deg_loss(pred)

        # compute image reconstruction loss
        img_recon_loss = F.mse_loss(pred["img_real"], batch["img"])

        # compute the total loss
        loss = (
            img_recon_loss
            + self.dcc_factor * dcc_loss
            + self.geom_factor * geometry_deg_loss
            + 0.1 * centering_loss
        )

        self.log("train/loss", loss)
        self.log("train/centering_loss", centering_loss)
        self.log("train/dcc_loss", dcc_loss)
        self.log("train/geometry_deg_loss", geometry_deg_loss)
        self.log("train/norm_shift", pred["out_norm"][0])
        self.log("train/img_loss", img_recon_loss)
        self.log("train/norm_scale", pred["out_norm"][1])

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        self.val_step_outputs.append((batch, pred))

    def on_validation_epoch_end(self):
        # self.val_step_outputs is a list of the batches and predictions of all calls (one per batch) to validation_step

        # collect all batches and predictions in two separate tuples
        # preds is now a tuple with length equal to the number of validation batches. Each element is equal to the return value of forward. Analogously for 'batches'.
        batches, preds = zip(*self.val_step_outputs)

        # clear the validation outputs list for the next epoch
        self.val_step_outputs.clear()

        # 'batches' is a tuple of dicts with tensors as values. Concatenate all tensors of the same key in the batch dimension. This is similar to torch.utils.data.default_collate, except that we do not stack but concat, since there is already a batch dimension.
        # gt stands for 'ground truth'
        gt = {}
        for k in ["img", "idx", "conf_idx"]:
            gt[k] = torch.cat([b[k] for b in batches], dim=0)

        # analogously for preds
        pred = {}
        for k in ["img_real", "conf", "transl"]:
            if preds[0][k] is not None:
                pred[k] = torch.cat([p[k] for p in preds], dim=0)

        # Add all media that should be logged to Wandb to this dict. The key is the name under which the value will be added to the dashboard. This dict will we logged using wandb.log() (https://docs.wandb.ai/ref/python/log)
        log_dict = {}

        gt_confs = self.dataset.gt_confs.to(self.device)
        c_alpha_metrics = compute_val_metrics(pred["conf"], gt_confs, gt["conf_idx"])

        self.log("val/RMSD", c_alpha_metrics["RMSD"].mean())
        self.log("val/EMD-RMSD", c_alpha_metrics["EMD-RMSD"])

        ### Log media

        # Log input and reconstructed images
        n_imgs = 50
        log_images_real(gt, pred, n_imgs, log_dict)

        # Log the predicted conformations as a 3D scatter plot
        n_plots = 30
        log_3d_reconstructions_synth_hetero(
            gt, pred, self.molecule_model.prior, gt_confs, log_dict
        )

        # Log bar chart with the distances between consecutive C-alpha atoms
        log_dcc_by_residue_barchart(pred, self.molecule_model, log_dict)

        # Log bar chart with mean predicted translation per residue
        log_mean_transl_by_residue_barchart(pred, log_dict)

        # Log bar chart with mean squared deviation by residue
        log_MSD_by_residue_barchart_hetero(gt_confs, gt, pred["conf"], log_dict)

        # Log all media objects at the current step. step=self.global_step ensures the steps are synced with the steps on the x-axes of all the other plots.
        self.logger.experiment.log(log_dict, step=self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer
