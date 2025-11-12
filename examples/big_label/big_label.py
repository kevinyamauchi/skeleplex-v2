"""Example of labeling an image lazily in parallel.

If you do not have cupy installed, you can change the backend to "cpu" to
run on the CPU instead.
"""

import napari
import numpy as np
import zarr

from skeleplex.skeleton import label_chunks_parallel, merge_touching_labels


def make_mock_segmentation(segmentation_path: str, chunk_shape: tuple = (8, 8, 8)):
    """Save a mock segmentation zarr array for testing."""
    array_shape = (20, 20, 20)

    # Create zarr array
    label_image = zarr.open(
        segmentation_path,
        mode="w",
        shape=array_shape,
        chunks=chunk_shape,
        dtype=np.uint16,
    )

    # Create an object spanning multiple chunks with different labels in each chunk
    # Object spans z=(5:18), which crosses chunk boundary at z=8 and z=16
    label_image[5:18, 2:4, 2:4] = 1

    # Create a non-touching object in a different location (label 4)
    label_image[10:15, 10:12, 10:12] = 4


if __name__ == "__main__":
    # path to save the mock segmentation zarr array
    segmentation_path = "segmentation.zarr/segmentation"

    # path to save the chunk-wise labeled zarr array
    chunkwise_labeled_path = "segmentation.zarr/chunkwise_labels"

    # path to save the final merged labeled zarr array
    final_labeled_path = "segmentation.zarr/final_labels"

    # number of processes to use for processing
    n_label_processes = 2
    n_merge_processes = 2

    # shape of the chunks
    chunk_shape = (8, 8, 8)

    # make the mock segmentation
    # this has two objects, one of which spans multiple chunks
    # and one that is in a single chunk
    make_mock_segmentation(segmentation_path, chunk_shape)

    # perform the chunkwise labeling in parallel
    max_label_value = label_chunks_parallel(
        input_path=segmentation_path,
        output_path=chunkwise_labeled_path,
        chunk_shape=chunk_shape,
        n_processes=n_label_processes,
        pool_type="fork",
        backend="cupy",
    )

    # merge touching labels across chunk boundaries
    merge_touching_labels(
        label_image_path=chunkwise_labeled_path,
        output_image_path=final_labeled_path,
        chunk_shape=chunk_shape,
        max_label_value=max_label_value,
        n_processes=n_merge_processes,
        pool_type="fork",
        backend="cupy",
    )

    # load the arrays
    segmentation = np.asarray(zarr.open(segmentation_path, mode="r"))
    chunkwise_labels = np.asarray(zarr.open(chunkwise_labeled_path, mode="r"))
    final_labels = np.asarray(zarr.open(final_labeled_path, mode="r"))

    # view the results
    viewer = napari.Viewer()
    viewer.add_labels(segmentation, name="segmentation", visible=False)
    viewer.add_labels(chunkwise_labels, name="chunkwise_labels", visible=False)
    viewer.add_labels(final_labels, name="final_labels", visible=True)

    viewer.dims.ndisplay = 3

    napari.run()
