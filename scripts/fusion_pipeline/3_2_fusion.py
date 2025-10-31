"""Fusion Part 3.2: Generate Final Skeleton from Optimal Tree."""
##################################################################################################
#                                           IMPORTS
##################################################################################################

import argparse
import time

import dask.array as da
from skimage.morphology import skeletonize as sk_skeletonize

from skeleplex.skeleton._skeletonize import threshold_skeleton

##################################################################################################
#                                           RUNNING FUSION
##################################################################################################
start_time1 = time.time()
# Define the image prefix used to name the files
image_prefix = "IMAGE_PREFIX"  # ADAPT HERE

# Example: define scales and their valid ranges
scale_ranges_manual = {
    0: (1, 5),
    -1: (5, 12),
    -2: (12, 30),
    -3: (30, 150),
}  # ADAPT HERE


# Example: define thresholds for the final fusion 3 step
thresholds_list = [0.4, 0.45, 0.55, 0.65, 0.7]  # ADAPT HERE

parser = argparse.ArgumentParser()
parser.add_argument(
    "--job-index", help="this is the index of the submitted job", type=int
)

args = parser.parse_args()


print("Scale the image here:")
threshold = thresholds_list[args.job_index]
threshold_name = int(threshold * 100)
print("Job Index: ", args.job_index)
print("Threshold: ", threshold)
print("Threshold name:", threshold_name)


############## from Fusion Part 1 ##############

# Load Scale Map
lung_image_scale_map = da.from_zarr(
    f"data/{image_prefix}_image_scale_map.zarr/scale_original"
)
lung_image_scale_map = lung_image_scale_map.rechunk((192, 192, 192))


################ Fusion Part 3 #################

# Threshold Image
start_time3 = time.time()
lung_image_optimum = da.from_zarr(f"data/{image_prefix}_optimal_tree.zarr")
lung_image_binary_skeleton = threshold_skeleton(lung_image_optimum, threshold=threshold)
lung_image_binary_skeleton.to_zarr(
    f"data/{image_prefix}_optimal_tree_binary_skeleton.zarr/ts{threshold_name}",
    overwrite=True,
)
print(f"--- Thresholding optimal tree took {time.time() - start_time3} seconds ---")

# Perform Thinning / Skeletonizing
start_time4 = time.time()
lung_image_binary_skeleton = lung_image_binary_skeleton.rechunk((192, 192, 192))
lung_image_final_skeleton = lung_image_binary_skeleton.map_overlap(
    sk_skeletonize, depth=10, boundary=0, dtype=lung_image_binary_skeleton.dtype
)

lung_image_final_skeleton = lung_image_final_skeleton.rechunk((192, 192, 192))
lung_image_final_skeleton.to_zarr(
    f"data/{image_prefix}_final_skeleton.zarr/ts{threshold_name}", overwrite=True
)

print(f"--- Skeletonizing optimal tree took {time.time() - start_time4} seconds ---")
