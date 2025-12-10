"""Make the demo tubes image to test the parallel inference."""

from pathlib import Path

import zarr

from skeleplex.data.tubes import draw_tubes_image_example
from skeleplex.skeleton.distance_field import local_normalized_distance

if __name__ == "__main__":
    # set to True to visualize the result with napari
    visualize_with_napari = True

    # path to save the demo image
    output_path = Path("demo_tubes.zarr")

    # chunk size for the zarr file
    chunk_size = (20, 20, 20)

    # create the demo tubes image
    tubes_image = draw_tubes_image_example().astype(bool)

    # compute the normalized distance field
    distance_field_image = local_normalized_distance(tubes_image)

    # save the tubes image with the specified chunk size
    tubes_path = output_path / "segmentation"
    tubes_zarr = zarr.open(
        tubes_path,
        mode="w",
        shape=tubes_image.shape,
        chunks=chunk_size,
        dtype=tubes_image.dtype,
    )
    tubes_zarr[:] = tubes_image

    # save the distance field image with the specified chunk size
    distance_field_path = output_path / "distance_field"
    distance_field_zarr = zarr.open(
        distance_field_path,
        mode="w",
        shape=distance_field_image.shape,
        chunks=chunk_size,
        dtype=distance_field_image.dtype,
    )
    distance_field_zarr[:] = distance_field_image

    if visualize_with_napari:
        import napari
        import numpy as np

        distance_field_image_zarr = zarr.open(distance_field_path, mode="r")
        tubes_image_zarr = zarr.open(tubes_path, mode="r")

        viewer = napari.Viewer()
        viewer.add_image(
            np.asarray(distance_field_image_zarr),
            name="distance_field",
            colormap="magma",
            contrast_limits=(0, 1),
        )
        viewer.add_labels(
            np.asarray(tubes_image_zarr),
            name="tubes_segmentation",
        )
        napari.run()
