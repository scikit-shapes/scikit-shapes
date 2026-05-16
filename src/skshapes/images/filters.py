import numpy as np
import scipy
import torch

from .. import Image, Mask, SparseImage
from .utils import eigenvalues, otsu_threshold

##############################
#### Convolution offsets #####
##############################


def cube_offsets(radius, device=None):
    window = torch.arange(-radius, radius + 1, device=device)
    return torch.stack(
        torch.meshgrid(window, window, window, indexing="xy"), dim=-1
    ).reshape(-1, 3)


def sphere_offsets(square_radius, device=None):
    pattern = cube_offsets(square_radius, device=device)
    return pattern[(pattern**2).sum(dim=-1) == square_radius]


def coarse_sphere_offsets(radius, device=None):
    pattern = cube_offsets(radius, device=device)
    pattern_sqradius = (pattern**2).sum(dim=-1)
    pattern_filter = torch.logical_and(
        pattern_sqradius.greater((radius - 1) ** 2),
        pattern_sqradius.less_equal(radius**2),
    )

    return pattern[pattern_filter], pattern_sqradius[pattern_filter]


def manhattan_offsets(device=None):
    return torch.tensor(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
        dtype=torch.int64,
        device=device,
    )


def onedim_gaussian_offsets(sigma, device=None):
    radius = int(np.ceil(3 * sigma))

    coords = torch.arange(
        -radius, radius + 1, dtype=torch.int64, device=device
    )
    pattern = torch.zeros(
        (2 * radius + 1, 3), dtype=torch.int64, device=device
    )
    pattern[:, 0] = coords

    weights = torch.exp(-0.5 * (coords / sigma) ** 2)
    weights = weights / weights.sum()

    return pattern, weights


###########################
#### Filter functions #####
###########################


def hysteresis_threshold(image: Image, *, low: float, high: float) -> Mask:
    """
    Computes the hysteresis threshold mask of the image, as in
    https://scikit-image.org/docs/0.25.x/api/skimage.filters.html#skimage.filters.apply_hysteresis_threshold.

    A first mask is computed by selecting pixels whose values are larger than the low threshold. The connected
    components of the mask are then computed, and only the ones that contain a pixel whose value is larger than the high
    threshold are conserved.

    Parameters
    ----------
    image
        The image to apply the threshold to.
    low
        Minimum value for a pixel to be taken in the mask.
    high
        Each connected component of the output mask should contain a pixel whose value is larger than this threshold.
    Returns
    -------
    Mask
        The hysteresis mask.

    """

    low_mask = image > low
    high_mask = image[low_mask] > high

    low_mask_components = low_mask.connected_components(
        offsets=manhattan_offsets(device=image.device)
    )
    valid_components = low_mask_components[high_mask]

    valid_components = valid_components.unique().output
    return low_mask_components.isin(valid_components[valid_components != 0])


def signed_distance_transform(
    mask: Mask, dilation: int = 0, maximum_radius: int = 10
) -> SparseImage:
    """Computes the signed distance transform of the input mask.

    The algorithm uses the fact that vessels have a small radius to compute the distance transform using a brute force
    approach.


    Parameters
    ----------
    mask
        The input mask on which the signed distance transform will be computed.

    dilation
        The maximum distance of outside pixels for which the signed distance should be computed.

    maximum_radius
        The maximum radius of a vessel in the image. The distance transform of inside pixels whose distance is greater
        than this radius will be set to maximum_radius.

    Returns
    -------
    MaskedImageData
        The signed distance transform of the input mask.

    Notes
    -----

    The positive (resp. negative) radius values are decreased (resp. increased) by 0.5.
    """

    def _outside_distance_transform(_mask: Mask) -> SparseImage:
        """
        Computes the negative part of the signed distance transform of the masked image.

        For each possible distance value (in decreasing order), the set of pixels at this distance of a mask pixel is
        computed, and the radius of all these pixels who does not belong to the mask are set to the current distance value.

        """
        output_image = SparseImage.zeros(
            shape=_mask.shape, dtype=torch.float32, device=_mask.device
        )

        for square_radius in range(dilation**2, 0, -1):
            offsets = sphere_offsets(square_radius, device=_mask.device)

            if offsets.shape[0] > 0:
                assignable = _mask.convolution(
                    offsets=offsets, kernel="any"
                ).difference(_mask)
                output_image[assignable] = 0.5 - np.sqrt(square_radius)

        return output_image

    def _inside_distance_transform(_mask: Mask) -> SparseImage:
        """
        Computes the positive part of the signed distance transform of the masked image.

        For each possible distance value (in increasing order), the set of pixels at this distance of a mask pixel is
        computed, and the radius of all the mask pixels who contains a neighbor pixel that is not in the mask are set to the
        current distance value.

        To reduce the number of iterations, all the distances between sqrt(radius ** 2 - 1) and radius are processed at the
        same time, and the exact radius to update is kept in the weights of the convolution. Once a distance is given to
        a mask pixel, it is marked as assigned and will not be processed in the further iterations.

        """
        output_image = SparseImage.fill(
            shape=_mask.shape,
            dtype=torch.float32,
            device=_mask.device,
            fill_value=maximum_radius + 0.5,
        )

        default_value = maximum_radius**2 + 1

        def kernel(u, w):
            return torch.min(torch.where(u, default_value, w), dim=-1)[0]

        for radius in range(1, maximum_radius + 1):
            offsets, weights = coarse_sphere_offsets(
                radius, device=_mask.device
            )

            new_radii = mask.masked_convolution(
                mask=_mask,
                offsets=offsets,
                weights=weights,
                kernel=kernel,
                outside_value=default_value,
            )
            assignable = new_radii != default_value
            output_image += new_radii.apply_pointwise(
                lambda x: torch.sqrt(x) - 0.5 - (maximum_radius + 0.5)
            )[assignable]

            _mask = _mask.difference(assignable)
            if _mask.n_points == 0:
                break

        return output_image[mask]

    return _inside_distance_transform(mask) + _outside_distance_transform(mask)


def gaussian_blur(image: SparseImage, sigma: float = 1.0) -> SparseImage:
    """
    Apply a gaussian blur to the input image.
    Parameters
    ----------
    image
    sigma

    Returns
    -------

    Notes
    -----
    The transformation is performed in-place.

    """

    blurred = image
    offsets, weights = onedim_gaussian_offsets(
        sigma=sigma, device=image.device
    )

    for _ in range(3):
        blurred = blurred.convolution(
            offsets=offsets, weights=weights, kernel="sum"
        )
        offsets = offsets[:, (2, 0, 1)]

    return blurred


def hessian(
    image: SparseImage, mask: Mask
) -> SparseImage:  # TODO repasser dessus
    diag_offsets = torch.tensor(
        [
            [[-1, 0, 0], [1, 0, 0], [0, 0, 0]],
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, -1], [0, 0, 1], [0, 0, 0]],
        ],
        device=image.device,
    ).permute(1, 0, 2)

    upper_offsets = torch.tensor(
        [
            [[-1, -1, 0], [1, 1, 0], [1, -1, 0], [-1, 1, 0]],
            [[-1, 0, -1], [1, 0, 1], [1, 0, -1], [-1, 0, 1]],
            [[0, -1, -1], [0, 1, 1], [0, 1, -1], [0, -1, 1]],
        ],
        device=image.device,
    ).permute(1, 0, 2)

    diag_weights = (
        torch.tensor([1, 1, -2], dtype=torch.float32, device=image.device)
        .repeat((3, 1))
        .permute(1, 0)
    )
    upper_weights = (
        torch.tensor(
            [0.25, 0.25, -0.25, -0.25],
            dtype=torch.float32,
            device=image.device,
        )
        .repeat((3, 1))
        .permute(1, 0)
    )
    hessian_diag = image.masked_convolution(
        mask=mask, offsets=diag_offsets, weights=diag_weights, kernel="sum"
    )._flat_values
    hessian_upper = image.masked_convolution(
        mask=mask, offsets=upper_offsets, weights=upper_weights, kernel="sum"
    )._flat_values

    hess_vals = torch.cat([hessian_upper, hessian_diag, hessian_upper], dim=1)[
        :, [3, 0, 1, 6, 4, 2, 7, 8, 5]
    ].reshape(-1, 3, 3)
    return SparseImage(
        flat_indices=mask._flat_indices, values=hess_vals, shape=mask.shape
    )


def image_otsu_threshold(image: SparseImage) -> float:
    """
    Computes the Otsu threshold of the values distribution (https://en.wikipedia.org/wiki/Otsu%27s_method).

    This is a copy of the code of sckikit-image, adapted for pytorch to improve performance:
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu

    Parameters
    ----------
    values
        The list of values to threshold.

    """
    counts, bin_edges = image.histogram(bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    weight1 = torch.cumsum(counts, dim=0)
    weight2 = torch.cumsum(counts.flip(dims=[0]), dim=0).flip(dims=[0])

    mean1 = torch.cumsum(counts * bin_centers, dim=0) / weight1
    mean2 = (
        torch.cumsum((counts * bin_centers).flip(dims=[0]), dim=0)
        / weight2.flip(dims=[0])
    ).flip(dims=[0])

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = torch.argmax(variance12)
    return bin_centers[idx]


def frangi_filter(
    image: SparseImage,
    alpha: float = 0.2,
    beta: float = 0.5,
    gamma: float = 0.5,
    smoothing: float = 1.0,
) -> Mask:
    """
    Computes the Frangi mask of the input image, following 'Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever,
    M. A. (1998). Multiscale vessel enhancement filtering.'

    At each voxel, given l1, l2, l3 the absolute value of the tree eigenvalues of hessian image at this pixel (in
    increasing value), the method first computes the following parameters:

    ra = l1 / sqrt(l2 * l3), rb = l2 / l3, s = sqrt(l1**2 + l2**2 + l3**2).

    Then, the vesselness value is defined as:

    vesselness_value = (1 - exp(- ra**2 / (2 * alpha ** 2))) * exp(- rb**2 / (2 * beta ** 2)) * (1 - exp(- s**2 / (2 * gamma ** 2)))

    A thresholding is finally performed to only retain the voxels with the largest vesselness value. In practice, we use
    an Otsu thresholding to choose the threshold.

    Parameters
    ----------
    image
        The image to compute the mask from.
    alpha
        Selectivity parameter for circular structures
    beta
        Selectivity parameter for elongate structures.
    gamma
        Selectivity parameter for foreground voxels.
    smoothing
        Radius for the gaussian blur performed as preprocessing of the Frangi method.

    Returns
    -------

    """

    def _vesselness(hess_value: torch.Tensor) -> torch.Tensor:
        eigs = eigenvalues(hess_value)
        indices = (
            torch.arange(eigs.shape[0]).reshape(-1, 1).expand(eigs.shape[0], 3)
        )
        eigs = eigs[indices, eigs.abs().argsort(dim=1)]

        ra = eigs[:, 1].abs() / (alpha * eigs[:, 2].abs() + 1e-20)
        rb = eigs[:, 0].abs() / (
            beta * torch.sqrt(eigs[:, 1].abs() * eigs[:, 2].abs() + 1e-20)
        )
        s = eigs.norm(dim=1) / gamma

        vesselness_value = (
            (1 - torch.exp(-0.5 * ra**2))
            * torch.exp(-0.5 * rb**2)
            * (1 - torch.exp(-0.5 * s**2))
        )
        return vesselness_value * torch.logical_and(
            eigs[:, 1] < 0, eigs[:, 2] < 0
        )

    blurred = gaussian_blur(image, sigma=smoothing)
    hess = hessian(image=blurred, mask=image != 0)

    vesselness_values = hess.apply_pointwise(_vesselness)

    # threshold = image_otsu_threshold(vesselness_values.to(device='cpu'))
    threshold = otsu_threshold(vesselness_values._flat_values)

    return vesselness_values > threshold


def skeletonize(mask: Mask) -> Mask:
    """
    Computes a 1-voxel wide skeletonized mask of the input mask, following the method of 'Palágyi, K., & Kuba, A. (1998).
    A 3D 6-subiteration thinning algorithm for extracting medial lines.'

    At each iteration of the algorithm, the thinning is performed in one of the six oriented 3D directions. The bounding
    pixels from this directions are candidates for deletion. If the neighborhood of a candidate falls in one of the
    valid patterns, the pixel is deleted. The procedure continues until no more pixel can be deleted.

    Parameters
    ----------
    mask
        The input mask to perform the skeletonization on.

    Returns
    -------
    Mask
        The skeletonized mask.

    """
    device = mask.device

    def _check_pattern(u):
        u = u.reshape(-1, 3, 3, 3)

        upper_count = u[:, :, :, 0].reshape(-1, 9).sum(dim=-1)
        lower_count = u[:, :, :, 1:].reshape(-1, 18).sum(dim=-1)

        mask_1 = (upper_count == 0) * u[:, 1, 1, 2] * (lower_count >= 3)

        mask_2 = u[:, 1, 1, 2] * (
            (u[:, :2, :, 0].reshape(-1, 6).sum(dim=-1) == 0) * u[:, 2, 1, 1]
            + (u[:, 1:, :, 0].reshape(-1, 6).sum(dim=-1) == 0) * u[:, 0, 1, 1]
            + (u[:, :, :2, 0].reshape(-1, 6).sum(dim=-1) == 0) * u[:, 1, 2, 1]
            + (u[:, :, 1:, 0].reshape(-1, 6).sum(dim=-1) == 0) * u[:, 1, 0, 1]
        )

        mask_3 = ~u[:, 1, 1, 0] * u[:, 1, 1, 2]
        mask_3 = mask_3 * (
            ~(u[:, 0, 0, 0] + u[:, 0, 1, 0] + u[:, 1, 0, 0])
            * u[:, 2, 1, 1]
            * u[:, 1, 2, 1]
            + ~(u[:, 2, 0, 0] + u[:, 1, 0, 0] + u[:, 2, 1, 0])
            * u[:, 1, 2, 1]
            * u[:, 0, 1, 1]
            + ~(u[:, 2, 2, 0] + u[:, 2, 1, 0] + u[:, 1, 2, 0])
            * u[:, 0, 1, 1]
            * u[:, 1, 0, 1]
            + ~(u[:, 0, 2, 0] + u[:, 1, 2, 0] + u[:, 0, 1, 0])
            * u[:, 1, 0, 1]
            * u[:, 2, 1, 1]
        )

        mask_4 = (upper_count == 1) * u[:, 1, 1, 2]
        mask_4 = mask_4 * (
            u[:, 2, 2, 0] * u[:, 2, 2, 1]
            + u[:, 0, 2, 0] * u[:, 0, 2, 1]
            + u[:, 0, 0, 0] * u[:, 0, 0, 1]
            + u[:, 2, 0, 0] * u[:, 2, 0, 1]
        )

        mask_5 = (upper_count == 0) * ~u[:, 1, 1, 2] * (lower_count >= 3)
        mask_5 = mask_5 * (
            (u[:, 0, :, 1:].reshape(-1, 6).sum(dim=-1) == 0) * u[:, 2, 1, 2]
            + (u[:, :, 0, 1:].reshape(-1, 6).sum(dim=-1) == 0) * u[:, 1, 2, 2]
            + (u[:, 2, :, 1:].reshape(-1, 6).sum(dim=-1) == 0) * u[:, 0, 1, 2]
            + (u[:, :, 2, 1:].reshape(-1, 6).sum(dim=-1) == 0) * u[:, 1, 0, 2]
        )

        mask_6 = (
            (upper_count == 0)
            * ~u[:, 1, 1, 2]
            * (u[:, :, :, 1:].reshape(-1, 18).sum(dim=-1) >= 3)
        )
        mask_6 = mask_6 * (
            ~(
                u[:, 0, 0, 1]
                + u[:, 0, 0, 2]
                + u[:, 1, 0, 1]
                + u[:, 1, 0, 2]
                + u[:, 0, 1, 1]
                + u[:, 0, 1, 2]
            )
            * u[:, 1, 2, 2]
            * u[:, 2, 1, 2]
            + ~(
                u[:, 1, 0, 1]
                + u[:, 1, 0, 2]
                + u[:, 2, 0, 1]
                + u[:, 2, 0, 2]
                + u[:, 2, 1, 1]
                + u[:, 2, 1, 2]
            )
            * u[:, 0, 1, 2]
            * u[:, 1, 2, 2]
            + ~(
                u[:, 2, 1, 1]
                + u[:, 2, 1, 2]
                + u[:, 2, 2, 1]
                + u[:, 2, 2, 2]
                + u[:, 1, 2, 1]
                + u[:, 1, 2, 2]
            )
            * u[:, 1, 0, 2]
            * u[:, 0, 1, 2]
            + ~(
                u[:, 1, 2, 1]
                + u[:, 1, 2, 2]
                + u[:, 0, 2, 1]
                + u[:, 0, 2, 2]
                + u[:, 0, 1, 1]
                + u[:, 0, 1, 2]
            )
            * u[:, 2, 1, 2]
            * u[:, 1, 0, 2]
        )

        return mask_1 + mask_2 + mask_3 + mask_4 + mask_5 + mask_6

    skeleton = mask.copy()

    window = torch.tensor([-1, 0, 1], device=device)
    offsets = torch.stack(
        torch.meshgrid(window, window, window, indexing="ij"), dim=-1
    ).reshape(-1, 3, 3, 3)
    offsets = offsets.to(device=device)

    borders = torch.tensor(
        [
            [[0, 0, -1]],
            [[0, 0, 1]],
            [[0, -1, 0]],
            [[0, 1, 0]],
            [[-1, 0, 0]],
            [[1, 0, 0]],
        ]
    )
    borders = borders.to(device=device, dtype=torch.int64)

    unchanged_borders = 0
    while unchanged_borders < 6:
        unchanged_borders = 0

        for direction, border in enumerate(borders):
            directed_pattern = offsets.clone()
            if direction in [2, 3]:
                directed_pattern = directed_pattern.transpose(1, 2)
            if direction in [4, 5]:
                directed_pattern = directed_pattern.transpose(0, 2)
            if direction in [1, 3, 5]:
                directed_pattern = torch.flip(directed_pattern, dims=[2])
            directed_pattern = directed_pattern.reshape(27, 3)

            border_pixels = skeleton.masked_convolution(
                offsets=border, kernel=lambda u: ~u.squeeze(-1)
            )
            deletable = skeleton.masked_convolution(
                mask=border_pixels,
                offsets=directed_pattern,
                kernel=_check_pattern,
            )

            if deletable.n_points == 0:
                unchanged_borders += 1
            else:
                skeleton = skeleton.difference(deletable)

    return skeleton


def skeleton_to_graph(mask: Mask) -> (np.ndarray, scipy.sparse.csr_matrix):
    window = torch.tensor([-1, 1, 0])
    offsets = (
        torch.stack(
            torch.meshgrid(window, window, window, indexing="ij"), dim=-1
        )
        .reshape(-1, 3)[:-1, :]
        .to(device=mask.device)
    )

    raw_adjmatrix = mask._adjacency_csr_matrix(offsets=offsets).tocoo()
    row, col, data = raw_adjmatrix.col, raw_adjmatrix.row, raw_adjmatrix.data

    junctions = np.argwhere(raw_adjmatrix.sum(axis=1) >= 3)
    junction_edges = np.logical_and(
        np.isin(row, junctions), np.isin(col, junctions)
    )

    row_junctions, col_junctions, data_junctions = (
        row[junction_edges],
        col[junction_edges],
        data[junction_edges],
    )
    junctions_adjmatrix = scipy.sparse.csr_matrix(
        (data_junctions, (row_junctions, col_junctions)),
        shape=(mask.n_points, mask.n_points),
    )

    junctions_mst = scipy.sparse.csgraph.minimum_spanning_tree(
        junctions_adjmatrix
    ).astype(bool)
    adjmatrix = (
        raw_adjmatrix - junctions_adjmatrix + (junctions_mst + junctions_mst.T)
    )

    pos = mask.indices.cpu().numpy()

    return pos, adjmatrix
