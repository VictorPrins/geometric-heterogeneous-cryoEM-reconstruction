import torch
import wandb
from PIL import Image
import plotly.graph_objects as go
from utils import to_grayscale_uint8
from evaluation import compute_minimum_SD


def log_images_real(gt, pred, n_imgs, log_dict):
    """
    Logs n_imgs pairs of cryo-em images with the input (ground truth) image on the left and the reconstructed image on the right.

    gt - dict with the full validation set in the batch dimension of all tensors.
    pred - dict with the predictions of the full validation set. Must be ordered in the batch dimension like gt.
    """

    # Get n_imgs from the whole range of ground truth images. This ensures we display projections of multiple underlying conformations.
    idx = torch.linspace(
        0, gt["img"].shape[0] - 1, n_imgs, dtype=torch.long, device=gt["img"].device
    )

    # get the images to be logged and their corresponding dataset indices
    gt_imgs = gt["img"][idx]
    pred_imgs = pred["img_real"][idx]
    indices = gt["idx"][idx].numpy(force=True)
    # imgs shape [n_imgs, side_len, side_len], indices shape [n_imgs]

    img_side_len = gt_imgs.shape[1]

    # put gt and pred side by side, split by a single white divider column
    divider = torch.zeros((n_imgs, img_side_len, 1)).to(gt_imgs)
    imgs = torch.cat([gt_imgs, divider, pred_imgs], dim=2)
    # shape [n_imgs, side_len, side_len*2 + 1]

    # normalize to [0,255] range and convert to numpy for PIL
    imgs = to_grayscale_uint8(imgs).numpy(force=True)

    # set the divider colum of pixels to full white
    imgs[:, :, img_side_len] = 255

    # We want to log all images to a panel named "image_reconstructions". To do so, we must put them in a list.
    log_dict["image_reconstructions"] = [
        wandb.Image(
            data_or_path=Image.fromarray(imgs[i], mode="L"),
            caption=f"Input (L) and reconstruction (R) of index {indices[i]}",
        )
        for i in range(n_imgs)
    ]


def log_3d_reconstructions_synth_hetero(gt, pred, prior_conf, gt_confs, log_dict):
    """
    Logs one reconstruction for every ground truth conformation as 3D scatter plots with one point per C-alpha atom.

    This method is for a heterogeneous synthetic dataset with uniform distribution

    gt - dict with the full validation set in the batch dimension of all tensors.
    pred - dict with the predictions of the full validation set.
    gt_confs - shape [n_confs, n_residues, 3] with the homogeneous ground truth conformation
    prior_conf - the prior conformation of shape [n_residues, 3]
    """
    fig = go.Figure()

    # the residue indices, for labeling the atoms in the figure
    res_id = torch.arange(prior_conf.shape[0], device=prior_conf.device)

    # add the prior conf to the figure
    prior_conf_cpu = prior_conf.numpy(force=True)
    fig.add_trace(
        go.Scatter3d(
            x=prior_conf_cpu[:, 0],
            y=prior_conf_cpu[:, 1],
            z=prior_conf_cpu[:, 2],
            customdata=res_id.numpy(force=True),
            legendrank=-1,
            marker_color="#2CA02C",
            name=f"prior conf",
            hovertemplate="res=%{customdata:d}",
        )
    )

    # add the ground truth of the first conformation to the plot
    gt_cpu = gt_confs.numpy(force=True)
    fig.add_trace(
        go.Scatter3d(
            x=gt_cpu[0, :, 0],
            y=gt_cpu[0, :, 1],
            z=gt_cpu[0, :, 2],
            customdata=res_id.numpy(force=True),
            legendrank=-1,
            marker_color="red",
            name=f"gt conf 0",
            hovertemplate="res=%{customdata:d}",
        )
    )

    # add the ground truth of the last conformation to the plot. So the extremes of the ground truth conformations are plotted, and the reconstructions should be somewhat uniformly distributed between these two.
    fig.add_trace(
        go.Scatter3d(
            x=gt_cpu[101, :, 0],
            y=gt_cpu[101, :, 1],
            z=gt_cpu[101, :, 2],
            customdata=res_id.numpy(force=True),
            legendrank=-1,
            marker_color="purple",
            name=f"gt conf 101",
            hovertemplate="res=%{customdata:d}",
        )
    )

    # we want to log every third conformation. So set one in three counters to 0.
    counter = {}
    for i in range(200):
        counter[i] = 1
        if i % 3 == 0:
            counter[i] = 0

    # loop over the whole validation set
    for i in range(gt["conf_idx"].shape[0]):
        conf_idx = int(gt["conf_idx"][i])

        # if we haven't yet plotted this conformation enough times, then plot the current reconstruction
        if counter[conf_idx] < 1:
            counter[conf_idx] += 1

            # get the predicted conformation
            pred_conf = pred["conf"][i]

            # get the ground truth conformation
            gt_conf = gt_confs[conf_idx]
            # all conformations are shape [n_residues, 3]

            # get the predicted atom-wise translation w.r.t the prior
            transl = torch.linalg.vector_norm(pred_conf - prior_conf, dim=-1)

            # get the atom-wise error w.r.t the ground truth conformation
            error = torch.linalg.vector_norm(pred_conf - gt_conf, dim=-1)

            # get the dataset index of this example
            data_idx = gt["idx"][i]

            # This contains arrays of size [n_residues], stacked in the last dimension. We use this to display additional info for each datapoint when hovering over it with the mouse. See argument 'hovertemplate'
            customdata = torch.stack([error, transl, res_id], dim=-1).numpy(force=True)

            pred_conf_cpu = pred_conf.numpy(force=True)

            # plot the predicted conformation
            fig.add_trace(
                go.Scatter3d(
                    x=pred_conf_cpu[:, 0],
                    y=pred_conf_cpu[:, 1],
                    z=pred_conf_cpu[:, 2],
                    customdata=customdata,
                    marker_color="darkblue",
                    legendrank=conf_idx,
                    name=f"pred (gt={conf_idx}, i={data_idx})",
                    hovertemplate="<br>".join(
                        [
                            "ε=%{customdata[0]:.1f}Å",
                            "Δx=%{customdata[1]:.1f}Å",
                            "res=%{customdata[2]:d}",
                            "(%{x:.1f}, %{y:.1f}, %{z:.1f})",
                        ]
                    ),
                )
            )

    # don't show any axes or background; only the datapoints
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.update_traces(marker_size=4, line_width=2)

    # Add to the log dict
    log_dict["3d_reconstructions"] = wandb.Plotly(fig)


def log_dcc_by_residue_barchart(pred, molecule_model, log_dict):
    """
    Logs a bar chart with on the x-axis each residue number (except for the last), and on the y-axis the distance in Angstrom between each C-alpha atom and the C-alpha atom of the residue to the right of it. This is called the DCC, or distance between consecutive C-alpha atoms.

    The values are averaged across the validation set, such that for each residue the y-value is the average DCC for this residue throughout the whole validation set. The standard deviation across the validation set is drawn as an error bar for each residue.

    The target DCC of 3.8 Angstrom is drawn as a line through the figure. The DCCs in the predictions should approach this line, since DCC values in actual proteins deviate very little from this value.

    pred - dict with the predictions of the full validation set.
    """
    # get the distances between consecutive C-alpha atoms for all predictions
    pred_dcc = molecule_model.dcc(pred["conf"])
    # shape [B, n_residues - 1]

    # reduce over all validation samples to shape [n_residues - 1]
    dcc_mean = pred_dcc.mean(dim=0)
    dcc_std = pred_dcc.std(dim=0)

    # Consecutive C-alpha atoms should be very close to 3.8 Angstrom apart (Chakraborty, Venkatramani, et al, 2013, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3892923/).
    dcc_target = 3.8 * torch.ones_like(dcc_mean)

    # x values are the residue indices [n_residues - 1]
    x_vals = torch.arange(dcc_mean.shape[0]).numpy(force=True)

    # round numbers and send all tensors to cpu
    dcc_mean = dcc_mean.round(decimals=2).numpy(force=True)
    dcc_std = dcc_std.round(decimals=2).numpy(force=True)
    dcc_target = dcc_target.numpy(force=True)

    y_min = 0
    y_max = 10

    fig = go.Figure()

    # plot the average C-alpha distances between residues, with the standard deviation across the validation set as error bar
    fig.add_trace(
        go.Bar(
            y=dcc_mean,
            x=x_vals,
            name="Validation set (± std)",
            marker_color="rgb(55,83,109)",
            # error_y=dict(type="data", array=dcc_std, width=0, thickness=0.8),
        )
    )

    # plot the target DCC of 3.8 as a line through the figure
    fig.add_trace(
        go.Scatter(
            y=dcc_target,
            x=x_vals,
            name="Target DCC",
            marker_color="red",
            line_width=1.5,
        )
    )

    fig.update_layout(
        title="Distance between consecutive C-alpha atoms by residue",
        title_x=0.5,
        yaxis_range=[y_min, y_max],
        yaxis_title_text="Å",
        legend=dict(
            x=0,
            y=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    log_dict["dcc_barchart"] = wandb.Plotly(fig)


def log_mean_transl_by_residue_barchart(pred, log_dict):
    """
    Logs a bar chart with on the x-axis each residue number and on the y-axis the mean translation (in Angstrom) of that residue from its prior position, with the mean taken across the validation set. The standard deviation across the validation set is drawn as an error bar.

    gt - dict with the full validation set in the batch dimension of all tensors.
    pred - dict with the predictions of the full validation set.
    """

    # get the predicted residue-wise translation w.r.t the prior
    transl = torch.linalg.vector_norm(pred["transl"], dim=-1)
    # shape [B, n_residues]

    # average and std across all samples
    transl_mean = transl.mean(dim=0)
    transl_std = transl.std(dim=0)
    # shape [n_residues]

    # x values are the residues numbers [n_residues]
    x_vals = torch.arange(transl_mean.shape[0]).numpy(force=True)

    # round numbers and send tensors to cpu
    transl_mean = transl_mean.round(decimals=2).numpy(force=True)
    transl_std = transl_std.round(decimals=2).numpy(force=True)

    fig = go.Figure()

    # plot the MSD averaged across the whole validation set
    fig.add_trace(
        go.Bar(
            y=transl_mean,
            x=x_vals,
            name="all",
            marker_color="rgb(55,83,109)",
            # error_y=dict(type="data", array=transl_std, width=0, thickness=0.8),
        )
    )

    fig.update_layout(
        title="Mean translation (± std) from prior conf by residue",
        title_x=0.45,
        yaxis_title_text="Å",
    )

    log_dict["mean_transl_by_residue_barplot"] = wandb.Plotly(fig)


def log_MSD_by_residue_barchart_hetero(gt_confs, gt, pred_conf, log_dict):
    """
    This is the heterogeneous version of this function

    Logs a bar chart with on the x-axis each residue number and on the y-axis the mean squared deviation (in Angstrom^2) of that residue from its ground truth position, with the mean taken across the validation set. The deviation is computed by first optimally aligning each conformation to its ground truth conformation, such that the total RMSD across the whole conformation is minimized, and then computing the deviation per residue. These deviations are then averaged across all validation samples. The standard deviation across the validation set is drawn as an error bar.

    gt_conf - Tensor of shape [n_confs, n_residues, 3] with all the ground truth conformations represented in the C-alpha model

    gt - dict with the full validation set in the batch dimension of all tensors

    pred_conf - Tensor of shape [N, n_residues, 3] with all the predicted conformations represend in the C-alpha model. The batch dimension must have the same size and order as gt.
    """

    # compute deviations between all atoms and their corresponding ground truth atom, after optimally aligning each conformation with the corresponding ground truth conformation
    sd = compute_minimum_SD(pred_conf, gt_confs[gt["conf_idx"]]).sqrt()
    # shape [n_samp, n_residues]

    # average and std across all samples
    sd_all_mean = sd.mean(dim=0)
    sd_all_std = sd.std(dim=0)
    # shape [n_residues]

    # x values are the residues numbers [n_residues ]
    x_vals = torch.arange(sd_all_mean.shape[0]).numpy(force=True)

    # round numbers and send tensors to cpu
    sd_all_mean = sd_all_mean.round(decimals=2).numpy(force=True)
    sd_all_std = sd_all_std.round(decimals=2).numpy(force=True)

    fig = go.Figure()

    # plot the MSD averaged across the whole validation set
    fig.add_trace(
        go.Bar(
            y=sd_all_mean,
            x=x_vals,
            name="validation set",
            marker_color="rgb(55,83,109)",
        )
    )

    fig.update_layout(
        title="Mean deviation from ground truth by residue",
        title_x=0.45,
        legend_title_text="Validation set slice",
        yaxis_title_text="Å",
    )

    log_dict["MSD_by_residue_barplot"] = wandb.Plotly(fig)
