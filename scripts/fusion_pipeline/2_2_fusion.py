"""Fusion Part 2.2: Calculate Distance Field on Scaled Images."""

##################################################################################################
#                                           IMPORTS
##################################################################################################
import argparse
import time

import dask
import dask.array as da
import numpy as np
import zarr

from skeleplex.skeleton.distance_field import local_normalized_distance_gpu

parser = argparse.ArgumentParser()


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

################ Fusion Part 2.2 ################

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


# Load scaled images form zarr to calculate distcance field on all scaled images
scaled_image = da.from_zarr(
    f"data/{image_prefix}_image_scaled.zarr/scale{scale_number}"
)
scaled_image = scaled_image.rechunk((192, 192, 192))

print("Scaled image was loaded and rechunked")

# Calculate Distance Field on all scaled images
print(" Calculate distance field next")
start_time = time.time()
with dask.config.set(num_workers=args.workers):
    dist_image = scaled_image.map_overlap(
        local_normalized_distance_gpu, depth=10, boundary=0, dtype=np.float32
    )

    # save as zarr
    dist_image = dist_image.rechunk((192, 192, 192))
    dist_image.to_zarr(
        f"data/{image_prefix}_distance_field_on_scales.zarr/scale{scale_number}",
        overwrite=True,
    )
    group = zarr.open_group(
        f"data/{image_prefix}_distance_field_on_scales.zarr", mode="a"
    )
    scale_factor = 2**scale_number
    group[f"scale{scale_number}"].attrs["scale"] = [
        scale_factor,
        scale_factor,
        scale_factor,
    ]

print(f"--- Calculating distance field on scale {
    scale_number} took {time.time() - start_time} seconds ---")
