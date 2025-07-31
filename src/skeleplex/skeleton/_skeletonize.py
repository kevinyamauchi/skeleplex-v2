from typing import Literal

import dask.array as da
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from morphospaces.networks.multiscale_skeletonization import (
    MultiscaleSkeletonizationNet,
)
from tqdm import tqdm

from skeleplex.skeleton._utils import get_skeletonization_model, make_image_5d


def skeletonize(
    image: np.ndarray,
    model: Literal["pretrained"] | MultiscaleSkeletonizationNet = "pretrained",
    roi_size: tuple[int, int, int] = (120, 120, 120),
    overlap: float = 0.5,
    stitching_mode: str = "gaussian",
    progress_bar: bool = True,
    batch_size: int = 1,
) -> np.ndarray:
    """Skeletonize a normalized distance field image.

    Parameters
    ----------
    image : np.ndarray
        The input image to skeletonize.
        This should be a normalized distance field image.
    model : Literal["pretrained"] | MultiscaleSkeletonizationNet = "pretrained",
        The model to use for prediction. This can either be an instance of
        MultiscaleSkeletonizationNet or the string "pretrained". If "pretrained",
        a pretrained model will be downloaded from the SkelePlex repository and used.
        Default value is "pretrained".
    roi_size : tuple[int, int, int]
        The size of each tile to predict on.
        The default value is (120, 120, 120).
    overlap : float
        The amount of overlap between tiles.
        Should be between 0 and 1.
        Default value is 0.5.
    stitching_mode : str
        The method to use to stitch overlapping tiles.
        Should be "gaussian" or "constant".
        "gaussian" uses a Gaussian kernel to weight the overlapping regions.
        "constant" uses equal weight across overlapping regions.
        "gaussian" is the default.
    progress_bar : bool
        Displays a progress bar during the prediction when set to True.
        Default is True.
    batch_size : int
        The number of tiles to predict at once.
        Default value is 1.
    """
    # add dim -> NCZYX
    expanded_image = torch.from_numpy(make_image_5d(image))

    # get the skeletonziation model if requested
    if model == "pretrained":
        model = get_skeletonization_model()

    # put the model in eval mode
    model.eval()

    # make the prediction
    with torch.no_grad():
        result = sliding_window_inference(
            inputs=expanded_image,
            sw_batch_size=batch_size,
            sw_device=torch.device("cuda"),
            predictor=model,
            roi_size=roi_size,
            overlap=overlap,
            mode=stitching_mode,
            device=torch.device("cpu"),
            progress=progress_bar,
        )

    # squeeze dims -> ZYX
    return torch.squeeze(torch.squeeze(result, dim=0), dim=0).numpy()


def skeletonize_chunkwise(
    input_dask_array: da.Array,
    model: str | MultiscaleSkeletonizationNet = "pretrained",
    chunk_size: tuple[int, int, int] = (512, 512, 512),
    roi_size: tuple[int, int, int] = (120, 120, 120),
    padding: tuple[int, int, int] = (60, 60, 60),
    overlap: float = 0.5,
    stitching_mode: str = "gaussian",
    batch_size: int = 1,
) -> da.Array:
    """
    Skeletonize a large volume chunk by chunk using SkelePlex's skeletonize().

    Parameters
    ----------
    input_dask_array : dask.array.Array
        Normalized distance field as a Dask array.
    model : str or MultiscaleSkeletonizationNet, optional
        The model to use for skeletonization.
        If "pretrained", a pretrained model will be used.
        Default is "pretrained".
    chunk_size : tuple
        Size of each chunk to process.
    roi_size : tuple
        ROI size for sliding window inference.
    padding : tuple
        Overlap margin in each direction.
    overlap : float
        Sliding window overlap for each chunk.
    stitching_mode : str
        Stitching mode for overlapping patches.
    batch_size : int
        Sliding window batch size.

    Returns
    -------
    dask.array.Array
        Skeleton prediction as a Dask array.
    """
    model = get_skeletonization_model()

    image_dask = input_dask_array
    image_shape = image_dask.shape

    start_indices = [
        range(0, s, cs) for s, cs in zip(image_shape, chunk_size, strict=False)
    ]
    depth_chunks = []

    for z_start in tqdm(start_indices[0], desc="Z"):
        height_chunks = []
        for y_start in start_indices[1]:
            width_chunks = []
            for x_start in start_indices[2]:
                z0 = max(z_start - padding[0], 0)
                y0 = max(y_start - padding[1], 0)
                x0 = max(x_start - padding[2], 0)

                z1 = min(z_start + chunk_size[0] + padding[0], image_shape[0])
                y1 = min(y_start + chunk_size[1] + padding[1], image_shape[1])
                x1 = min(x_start + chunk_size[2] + padding[2], image_shape[2])

                padded_chunk = image_dask[z0:z1, y0:y1, x0:x1].compute()

                predicted_skeleton = skeletonize(
                    padded_chunk,
                    model=model,
                    roi_size=roi_size,
                    overlap=overlap,
                    stitching_mode=stitching_mode,
                    progress_bar=False,
                    batch_size=batch_size,
                )

                crop_z0 = z_start - z0
                crop_y0 = y_start - y0
                crop_x0 = x_start - x0

                crop_z1 = crop_z0 + min(chunk_size[0], image_shape[0] - z_start)
                crop_y1 = crop_y0 + min(chunk_size[1], image_shape[1] - y_start)
                crop_x1 = crop_x0 + min(chunk_size[2], image_shape[2] - x_start)

                cropped_skeleton_prediction = predicted_skeleton[
                    crop_z0:crop_z1,
                    crop_y0:crop_y1,
                    crop_x0:crop_x1,
                ]

                skeleton_chunk = da.from_array(
                    cropped_skeleton_prediction,
                    chunks=cropped_skeleton_prediction.shape,
                )
                width_chunks.append(skeleton_chunk)

            height_chunks.append(da.concatenate(width_chunks, axis=2))
        depth_chunk = da.concatenate(height_chunks, axis=1)
        depth_chunks.append(depth_chunk)

    skeleton_prediction = da.concatenate(depth_chunks, axis=0)
    return skeleton_prediction


def threshold_skeleton(skeleton: da.Array, threshold: float) -> da.Array:
    """
    Threshold a skeleton prediction to produce a binary skeleton mask.

    Parameters
    ----------
    skeleton : dask.array.Array
        Input skeleton prediction array (e.g. probabilities).
    threshold : float
        Threshold value to binarize the skeleton prediction.

    Returns
    -------
    dask.array.Array
        Binary skeleton mask as uint8.
    """
    return (skeleton > threshold).astype("uint8")


def filter_skeleton_by_segmentation(
    skeleton: da.Array,
    segmentation: da.Array,
) -> da.Array:
    """
    Mask a predicted skeleton using a segmentation mask.

    Parameters
    ----------
    skeleton : dask.array.Array
        Skeleton prediction image (usually float-valued).
    segmentation : dask.array.Array
        Binary segmentation mask with same shape as skeleton.

    Returns
    -------
    dask.array.Array
        Masked skeleton image (same dtype as skeleton).
    """
    if skeleton.shape != segmentation.shape:
        raise ValueError("Skeleton and segmentation shapes do not match.")

    return skeleton * segmentation.astype(np.float32)
