"""Skeletonize the segmentation."""

import dask.array as da
import numpy as np
from skimage.morphology import skeletonize

if __name__ == "__main__":
    # set to True to visualize with napari
    visualize = True

    # path to the input segmentation
    segmentation_path = "bifurcating_tree.zarr"

    segmentation_image = da.from_zarr(segmentation_path)

    lung_image_final_skeleton = segmentation_image.map_overlap(
        skeletonize, depth=10, boundary=0, dtype=np.uint8
    )

    lung_image_final_skeleton.to_zarr("skeleton.zarr", overwrite=True)

    if visualize:
        import napari

        viewer = napari.Viewer()
        viewer.add_labels(lung_image_final_skeleton.compute())
        napari.run()
