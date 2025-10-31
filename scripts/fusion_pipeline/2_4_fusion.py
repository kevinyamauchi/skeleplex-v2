"""Fusion Part 2.4: Rescale Skeleton Predictions to Original Size."""

##################################################################################################
#                                           IMPORTS
##################################################################################################
import argparse
import time
from typing import Any

import dask
import dask.array as da
import numpy as np
import skimage.transform
import zarr


def resize(
    image: da.Array, output_shape: tuple[int, ...], *args: Any, **kwargs: Any
) -> da.Array:
    """
    Resize function.

    Adapted from:
    https://github.com/ome/ome-zarr-py/blob/5c5b45e46e468a3f582a583c915b9eefb636b82c/ome_zarr/dask_utils.py#L11C1-L62C63

    Wrapped copy of "skimage.transform.resize"
    Resize image to match a certain size.

    Parameters
    ----------
    image: da.Array
        Input image (dask array) to be resized.
    output_shape: tuple[int, ...]
        The shape that the resized output image should have.
    *args: Any
        Additional positional arguments to pass to skimage.transform.resize.
    **kwargs: Any
        Additional keyword arguments to pass to skimage.transform.resize.

    Returns
    -------
    da.Array
        Input arrays scaled to the specified scales.
    """
    factors = np.array(output_shape) / np.array(image.shape).astype(float)
    # Rechunk the input blocks so that the factors achieve an output
    # blocks size of full numbers.
    better_chunksize = tuple(
        np.maximum(1, np.round(np.array(image.chunksize) * factors) / factors).astype(
            int
        )
    )
    image_prepared = image.rechunk(better_chunksize)

    # If E.g. we resize image from 6675 by 0.5 to 3337, factor is 0.49992509 so each
    # chunk of size e.g. 1000 will resize to 499. When assumbled into a new array, the
    # array will now be of size 3331 instead of 3337 because each of 6 chunks was
    # smaller by 1. When we compute() this, dask will read 6 chunks of 1000 and expect
    # last chunk to be 337 but instead it will only be 331.
    # So we use ceil() here (and in resize_block) to round 499.925 up to chunk of 500
    block_output_shape = tuple(
        np.ceil(np.array(better_chunksize) * factors).astype(int)
    )

    # Map overlap
    def resize_block(image_block: da.Array, block_info: dict) -> da.Array:
        # if the input block is smaller than a 'regular' chunk (e.g. edge of image)
        # we need to calculate target size for each chunk...
        chunk_output_shape = tuple(
            np.ceil(np.array(image_block.shape) * factors).astype(int)
        )
        return skimage.transform.resize(
            image_block, chunk_output_shape, *args, **kwargs
        ).astype(image_block.dtype)

    output_slices = tuple(slice(0, d) for d in output_shape)
    output = da.map_blocks(
        resize_block, image_prepared, dtype=image.dtype, chunks=block_output_shape
    )[output_slices]
    return output.rechunk(image.chunksize).astype(image.dtype)


##################################################################################################
#                                           FUNCTIONS
##################################################################################################


def re_scale_image(scale_number: int, int_order: int, og_img: np.ndarray) -> np.ndarray:
    """
    Scale the image for the fusion algorithm.

    The image will be scaled back to the original size.

    Previously, the image has been scaled to the powers of two defined
    in the scale ranges dictionary. The scale ranges dictionary should
    therefore contain the exponent as the key.
    Rescaling is performed based on the reciprocal value of the
    previously used scale factor.

    (rescale-factor = 1/scale-factor = 1/(2**key))

    Parameters
    ----------
    scale_number : int
        This is a value that is used to map the scales to the radii in the radius_map.
    int_order : int
        This defines the interpolation order used for the resizing.
    og_img : np.ndarray
        Original lung segmentation image.

    Returns
    -------
    np.ndarray
        Arrays that have been re-scaled to the size they were
        before the first scaling step.
    """
    image = da.from_zarr(
        f"data/{image_prefix}_skeleton_predictions_on_scales.zarr/scale{scale_number}"
    )
    re_scale = 2**scale_number
    print(re_scale)

    dask_arr = resize(image, order=int_order, output_shape=(og_img.shape))

    # save as zarr

    filtered_dask_arr = dask_arr
    filtered_dask_arr.to_zarr(
        f"data/{image_prefix}_skeletonized_rescaled.zarr/origin_scale{scale_number}",
        overwrite=True,
    )
    group = zarr.open_group(f"data/{image_prefix}_skeletonized_rescaled.zarr", mode="a")
    group[f"origin_scale{scale_number}"].attrs["re_scaled_from"] = [
        re_scale,
        re_scale,
        re_scale,
    ]


##################################################################################################
#                                           RUNNING FUSION
##################################################################################################

# Define the image prefix used to name the files
image_prefix = "IMAGE_PREFIX"  # ADAPT HERE


# Example: define scales and their valid ranges
scale_ranges_manual = {
    0: (1, 5),
    -1: (5, 12),
    -2: (12, 30),
    -3: (30, 150),
}  # ADAPT HERE


# Load the initial image (here: label)
lung_image = da.from_zarr(f"data/{image_prefix}.zarr")  # ADAPT HERE


################ Fusion Part 2.4 ################
parser = argparse.ArgumentParser()
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


# Load skeletonized image to rescle back to original size
filtered_skeleton_img = da.from_zarr(
    f"data/{image_prefix}_skeleton_predictions_on_scales.zarr/scale{scale_number}"
)
filtered_skeleton_img = filtered_skeleton_img.rechunk((192, 192, 192))

scaled_image = da.from_zarr(
    f"data/{image_prefix}_image_scaled.zarr/scale{scale_number}"
)
scaled_image = scaled_image.rechunk((192, 192, 192))


print("Skeleton Prediction was saved, loaded and rechunked")
# Rescale skeletonized images backl to original size
# Scale the image
print("Re-scale image next")
start_time2 = time.time()
lung_image = lung_image.rechunk((96, 96, 96))
with dask.config.set(num_workers=args.workers):
    re_scale_image(scale_number=scale_number, int_order=1, og_img=lung_image)

print(f"--- Re-scaling took {time.time() - start_time2} seconds ---")
print("Image was re-scaled. End of fusion part 2.")
