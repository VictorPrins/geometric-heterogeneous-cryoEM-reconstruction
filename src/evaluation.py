import torch
from torchtyping import TensorType
from tqdm import tqdm
import lap


def compute_val_metrics(
    sampled_confs: TensorType["n_samp", "n_atoms", 3],
    gt_confs: TensorType["n_gt", "n_atoms", 3],
    gt_conf_idx: TensorType["n_samp"],
):
    """
    Computes the validation metrics EMD-RMSD, RMSD, and accuracy.

    For all metrics, RMSD between any two point clouds is computed after optimally rigidly aligning the two point clouds, to find the minimum RMSD between them.

    EMD-RMSD is explained in section 4 of (Rosenbaum et al, 2021) (https://arxiv.org/abs/2106.14108). It is the mean RMSD over all n_samp pairs, after finding an optimal one-to-one pairing between each sample from the network distribution, and equally many samples from the ground truth distribution. Unlike in the Rosenbaum paper, the ground truth conformations need not actually be sampled; we know the actual ground truth distribution after all. Instead, all distinct ground truth conformations are passed to this function, and a uniform distribution over them is assumed. This metric measures how similar the reconstructed distribution as a whole is to the ground truth distribution as a whole.

    RMSD is a tensor of size [n_samp] with the RMSDs between each reconstructed conformation and its corresponding ground truth conformation. This metric measures how closely the network can reconstruct the ground truth conformation based on a single image. This metric can be averaged to get a summary statistic for the whole validation set.

    Accuracy is the percentage of all reconstructed conformations that were closest (in terms of RMSD) to their corresponding ground truth conformation, out of all ground truth conformations. This metric is closely related to RMSD. RMSD measures the distance to the ground truth conformation as a floating point number. Accuracy (for a single sample) is binary; either the reconstruction was closest to the ground truth conformation, or it was closest to another conformation in the datset.

    Pairwise RMSD is a tensor of shape [n_samp, n_gt] with the RMSD between each pair of predicted conformation vs ground truth conformation. The rows of the matrix are ordered like the argument sampled_confs to this function, and the columns like gt_confs.

    sampled_confs - conformations generated by the network from a random sample of input images from the training set

    gt_confs - all distinct ground truth conformations in the dataset. A uniform distribution over all ground truth conformations is assumed. I.e. p(conf) = 1/n_gt

    gt_conf_idx - for each reconstructed sample conformation, the index of the ground truth conformation in gt_confs. I.e. the gt conformation of sampled_confs[i] is gt_confs[gt_conf_idx[i]].
    """

    device = sampled_confs.device

    n_samp = sampled_confs.shape[0]
    n_gt = gt_confs.shape[0]

    # this function only works if we sample at least more times than there are ground truth conformations. For accurate EMD-RMSD computation it should be many times more
    assert n_samp > n_gt

    # Make all pairs (i,j) for i in {0, ..., n_samp-1} and j in {0, ..., n_gt-1}
    idx = torch.cartesian_prod(torch.arange(n_samp), torch.arange(n_gt)).to(device)
    # idx.shape[0] == N * n_conf. How to use: (A[idx[i,0]], B[idx[i,1]])

    # compute_minimum_sd() can reasonably handle tensors with 240M elements (approximately). Memory limitations slow it down significantly (at least on my laptop) for bigger tensors. So we batch it accordingly.

    max_els_per_batch = int(240 * 1e6)
    els_per_pointcloud = gt_confs[0].numel()
    batch_size = max_els_per_batch // els_per_pointcloud

    # chunk the indices in chunks of size batch_size
    idx_chunks = torch.split(idx, batch_size, dim=0)

    # matrix containing all pairwise RMSDs between the point clouds in sampled_confs and gt_confs
    pairwise_rmsd_mat = torch.zeros((n_samp, n_gt), device=device)

    # compute minimum rmsd between all pairs (forward the pairs in batches)
    for idx in tqdm(idx_chunks):
        sample_idx, gt_idx = idx.T
        # both of shape [batch_size]

        sd = compute_minimum_SD(sampled_confs[sample_idx], gt_confs[gt_idx])
        # shape [batch_size, n_points]
        # compute the RMSD per point cloud by averaging across the points and taking the sqrt
        rmsd = sd.mean(dim=1).sqrt()
        # shape [batch_size]

        # store the computed RMSDs in the matrix
        pairwise_rmsd_mat[sample_idx, gt_idx] = rmsd

    ### START Compute accuracy
    # find the index of the ground truth conformation that has the lowest RMSD with the sampled conformation - shape [n_samp]
    closest_gt_conf_idx = pairwise_rmsd_mat.argmin(dim=1)
    closest_is_actual_conf = closest_gt_conf_idx == gt_conf_idx

    # accuracy is the percentage of reconstructed conformations for which the closest conformation (out of all conformations in the dataset) was the actual ground truth conformation that was used for this prediction
    accuracy = torch.sum(closest_is_actual_conf) / closest_is_actual_conf.shape[0]
    ### END Compute accuracy

    ## START Compute RMSD
    # Get the RMSD between each sample and its corresponding ground truth conformation
    RMSD = pairwise_rmsd_mat[torch.arange(n_samp), gt_conf_idx]
    ## END Compute RMSD

    ## START Compute EMD-RMSD
    # Because each sampled conformation should be (optimally) one-to-one paired with a ground truth conformation (this is done by linear sum assignment), the cost matrix must be square. Therefore, we duplicate the columns until the matrix is square
    rep = n_samp // n_gt + 1
    cost_matrix = pairwise_rmsd_mat.repeat((1, rep))
    # truncate the last columns if n_samp is not divisible by n_gt
    cost_matrix = cost_matrix[:, :n_samp]

    # NOTE: the linear sum assignment is the limiting factor in terms of complexity of this funtion. It scales O(n^3). For a cost matrix of 2000x2000 it runs in ~0.5 seconds, and for 5000x5000 in 7 seconds (almost exactly as per O(n^3)). Take this into consideration when determining the validation set size. I tested multiple LAP solvers, and this one (source: https://github.com/gatagat/lap) was the fastest by a margin. It consistently ran 4 times quicker than the Scipy equivalent (and found the same solution) and another implementation I found on Github.

    # cast the cost matrix to Numpy for the LAP solver. The function returns the minimum total assignment cost (=sum of RMSDs) of matching n_samp (predicted conf, ground truth conf) pairs.
    cost, _, _ = lap.lapjv(cost_matrix.numpy(force=True))

    # the EMD-RMSD is the average cost per prediction
    EMD_RMSD = cost / n_samp

    return {
        "EMD-RMSD": EMD_RMSD,
        "RMSD": RMSD,
        "accuracy": accuracy,
        "pairwise_RMSD": pairwise_rmsd_mat,
    }


def compute_minimum_SD(A, B) -> TensorType["N", "n_points"]:
    """
    Computes the minimum total squared deviation for each pair of point clouds (A[i], B[i]) after optimally aligning A[i] to B[i]. The optimal rigid transformation (not allowing reflections) is found using Kabsch algorithm. See http://nghiaho.com/?page_id=671 for an explanation. It returns the squared deviation, rather than the RMSD.

    A and B of shape [N, n_points, 3], both containing N point clouds
    """

    # Compute centered versions of A and B by subtracting their centroid
    A_c = A - A.mean(dim=1, keepdim=True)
    B_c = B - B.mean(dim=1, keepdim=True)

    # Covariance matrix
    H = torch.bmm(A_c.transpose(1, 2), B_c)
    # shape [N, 3, 3]

    # SVD outputs the V matrix transposed
    U, _, Vt = torch.linalg.svd(H)
    # both of shape [N, 3, 3]

    # Rotation matrix
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))
    # shape [N, 3, 3]

    # if the determinant of the rotation matrix is negative, then R performs a reflection in addition to a rotation. We don't allow alignment using reflection as we want to treat reflected point clouds as different cases. To undo the reflection, we must negate the last row of Vt and redo the rotation matrix calculation
    mask = torch.linalg.det(R) < 0
    # shape [N]

    U = U[mask]
    Vt = Vt[mask]
    Vt[:, 2, :] *= -1
    R[mask] = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))
    # all rotation matrices in R are now purely rotation, without reflection

    # apply the optimal rotation to A_c
    A_c_rotated = torch.bmm(R, A_c.transpose(1, 2)).transpose(1, 2)

    # A_c_rotated and B_c are now positioned such that their RMSD is minimised; i.e. no rigid transformation exists that would further reduce the RMSD between them.

    # compute the squared deviation between A_c_rotated and B_c
    sd = (A_c_rotated - B_c).square().sum(dim=2)  # .mean(dim=1).sqrt()
    # shape [N, n_points]

    return sd