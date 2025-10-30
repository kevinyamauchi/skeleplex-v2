"""Fusion Part 2.3: Predict Skeletons on Scaled Distance Field Images."""

##################################################################################################
#                                           IMPORTS
##################################################################################################
import argparse
import copy
import gc
import itertools
import time
import uuid
from typing import Literal

import dask
import dask.array as da
import numpy as np
import torch
import zarr
from morphospaces.networks.multiscale_skeletonization import (
    MultiscaleSkeletonizationNet,
)

from skeleplex.skeleton._utils import get_skeletonization_model, make_image_5d

parser = argparse.ArgumentParser()


##################################################################################################
#                                           FUNCTIONS
##################################################################################################


def skeletonize_for_dask_simplified(
    image: np.ndarray,
    model: Literal["pretrained"] | MultiscaleSkeletonizationNet = "pretrained",
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
    """
    if "result_numpy" in locals():
        result_numpy = None
        del result_numpy
        gc.collect()
        torch.cuda.empty_cache()

    start_time = time.time()
    worker_id = str(uuid.uuid4())
    # add dim -> NCZYX
    expanded_image = torch.from_numpy(make_image_5d(image))

    # get the skeletonziation model if requested
    if model == "pretrained":
        model = get_skeletonization_model()

    # put the model in eval mode
    model.eval()

    # make the prediction
    with torch.no_grad():
        result = model(expanded_image.cuda())

    # add the code to squeeze the extra dims and convert to numpy here
    # squeeze dims -> ZYX
    skel_pred_torch = torch.squeeze(torch.squeeze(result, dim=0), dim=0)
    skel_pred = copy.deepcopy(skel_pred_torch)
    result_numpy = skel_pred.numpy(force=True)
    print(f"--- Skeleton Prediction for this chunk took {time.time()
                                                         - start_time} seconds ---")
    print(
        f"--- Worker: {worker_id}, Allocated Memory: {
            torch.cuda.memory_allocated(0)}"
    )
    # add the mode to clear the memory, e.g., delete the variables, clear cache, etc.
    start_time = time.time()
    del result, model, expanded_image, skel_pred_torch, skel_pred
    gc.collect()
    torch.cuda.empty_cache()
    print(f"-- Deleting result and model, empty cache took {time.time()
                                                             - start_time} seconds --")
    print(
        f"--- Worker: {worker_id}, Allocated Memory: {
            torch.cuda.memory_allocated(0)}"
    )
    return result_numpy


##################################################################################################
#                                           RUNNING FUSION
##################################################################################################

# Define the image prefix used to name the files
image_prefix = "IMAGE_PREFIX"  # ADAPT HERE

scale_ranges_manual = {
    0: (1, 5),
    -1: (5, 12),
    -2: (12, 30),
    -3: (30, 150),
}  # ADAPT HERE

################ Fusion Part 2.3 ################

parser.add_argument(
    "--job-index", help="this is the index of the submitted job", type=int
)
parser.add_argument(
    "--job-index-offset",
    help="this number assists in getting the negative and positive scales "
    "required for the fusion algorithm (submit a positive integer)",
    type=int,
)
parser.add_argument("--workers", help="this sets the number of workers", type=int)
args = parser.parse_args()


print("Scale the image here:")
scale_number = args.job_index - args.job_index_offset
print("Job Index: ", args.job_index)
print("Job Index Offset: ", args.job_index_offset)
print("Scale number: ", scale_number)


# Load scaled distance images form zarr to Predict Skeleton on all scaled images
dist_image = da.from_zarr(
    f"data/{image_prefix}_distance_field_on_scales.zarr/scale{scale_number}"
)
dist_image = dist_image.rechunk((192, 192, 192))


# Predict Skeleton on all scaled images
start_time = time.time()
with dask.config.set(num_workers=args.workers):
    # Zarr store
    save_here = zarr.open(
        f"data/{image_prefix}_skeleton_predictions_on_scales_chunkwise.zarr/scale{scale_number}",
        mode="w",
        shape=dist_image.shape,
        chunks=dist_image.chunks,
        dtype=dist_image.dtype,
    )

    for inds in itertools.product(*map(range, dist_image.blocks.shape)):
        chunk = dist_image.blocks[inds]
        chunk_mem = chunk.compute()
        dask_arr_chunk = skeletonize_for_dask_simplified(chunk_mem)

        region = tuple(
            slice(
                inds[dim] * dist_image.chunks[dim][0],
                inds[dim] * dist_image.chunks[dim][0] + dask_arr_chunk.shape[dim],
            )
            for dim in range(dist_image.ndim)
        )

        # save as zarr
        save_here[region] = dask_arr_chunk
        del dask_arr_chunk
        del region
        del chunk
        del chunk_mem
        gc.collect()

    group = zarr.open_group(
        f"data/{image_prefix}_skeleton_predictions_on_scales_chunkwise.zarr", mode="a"
    )
    scale_factor = 2**scale_number
    group[f"scale{scale_number}"].attrs["scale"] = [
        scale_factor,
        scale_factor,
        scale_factor,
    ]
print(f"--- Skeleton Prediction took {time.time()
                                      - start_time} seconds ---")
