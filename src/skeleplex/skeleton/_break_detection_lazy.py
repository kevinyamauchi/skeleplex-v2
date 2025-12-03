"""Functions for lazy chunk-based skeleton break repair."""

from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm

from skeleplex.skeleton._break_detection import repair_breaks
from skeleplex.utils import calculate_expanded_slice


def repair_breaks_chunk(
    skeleton: zarr.Array,
    output_skeleton: zarr.Array,
    segmentation: zarr.Array,
    expanded_slice: tuple[slice, slice, slice],
    actual_border: tuple[int, int, int],
    repair_radius: float,
) -> None:
    """Process a single chunk for skeleton break repair.

    Loads a chunk with boundary region, applies break repair only to endpoints
    in the core region, and writes the full result back to output. This allows
    repairs to extend into the boundary region while preventing duplicate
    processing of endpoints.

    Parameters
    ----------
    skeleton : zarr.Array
        The input zarr array containing the skeleton to repair.
    output_skeleton : zarr.Array
        The output zarr array to write the repaired skeleton to.
    segmentation : zarr.Array
        The zarr array containing the segmentation mask.
    expanded_slice : tuple[slice, slice, slice]
        Slice defining the chunk+boundary region to load from input.
    actual_border : tuple[int, int, int]
        The actual border size used (z, y, x). May be smaller than requested
        at volume edges.
    repair_radius : float
        The maximum Euclidean distance for connecting endpoints.

    Returns
    -------
    None
        Modifies output_zarr in-place.
    """
    # Load chunk+boundary data
    skeleton_chunk = np.array(skeleton[expanded_slice])
    segmentation_chunk = np.array(segmentation[expanded_slice])

    # Calculate endpoint bounding box within the loaded chunk
    # This restricts endpoint search to the core region
    chunk_shape = skeleton_chunk.shape
    endpoint_bbox = (
        (actual_border[0], actual_border[1], actual_border[2]),
        (
            chunk_shape[0] - actual_border[0],
            chunk_shape[1] - actual_border[1],
            chunk_shape[2] - actual_border[2],
        ),
    )

    # Apply repair to full chunk but only search for endpoints in core
    repaired_chunk = repair_breaks(
        skeleton_image=skeleton_chunk,
        segmentation=segmentation_chunk,
        repair_radius=repair_radius,
        endpoint_bounding_box=endpoint_bbox,
    )

    # Write full result (core + boundary) to output
    # This ensures repairs extending into boundary are captured
    output_skeleton[expanded_slice] = repaired_chunk


def repair_breaks_lazy(
    skeleton_path: str | Path,
    segmentation_path: str | Path,
    output_path: str | Path,
    repair_radius: float = 10.0,
    chunk_shape: tuple[int, int, int] = (256, 256, 256),
) -> None:
    """Repair breaks in a skeleton using lazy chunk-based processing.

    Processes a skeleton image that is too large to fit in memory by dividing
    it into chunks with overlapping boundaries. Each chunk is processed serially
    to avoid write conflicts. Endpoints are only searched in the core region of
    each chunk, while repairs can extend into the boundary regions.

    Parameters
    ----------
    skeleton_path : str or Path
        Path to the input zarr array containing the skeleton.
    segmentation_path : str or Path
        Path to the zarr array containing the segmentation mask.
    output_path : str or Path
        Path where the output zarr array will be created.
    repair_radius : float, default=10.0
        The maximum Euclidean distance for connecting endpoints.
        Also used as the boundary size around each chunk.
        The boundary is set to ceil(repair_radius) + 2 voxels.
    chunk_shape : tuple[int, int, int]
        The shape of each core chunk to process (z, y, x).
        Independent of the zarr storage chunk size.
        Default is (256, 256, 256).

    Returns
    -------
    None
        Creates a new zarr array at output_path with the repaired skeleton.

    Raises
    ------
    ValueError
        If input and segmentation shapes don't match.
    ValueError
        If repair_radius is not positive.
    ValueError
        If chunk_shape doesn't have exactly 3 elements.
    """
    # Validate inputs
    if repair_radius <= 0:
        raise ValueError(f"repair_radius must be positive, got {repair_radius}")

    if len(chunk_shape) != 3:
        raise ValueError(
            f"chunk_shape must be a 3-tuple, got length {len(chunk_shape)}"
        )

    # Convert paths to Path objects
    skeleton_path = Path(skeleton_path)
    segmentation_path = Path(segmentation_path)
    output_path = Path(output_path)

    # Open input arrays
    input_zarr = zarr.open(str(skeleton_path), mode="r")
    segmentation_zarr = zarr.open(str(segmentation_path), mode="r")

    # Validate shapes match
    if input_zarr.shape != segmentation_zarr.shape:
        raise ValueError(
            f"Input and segmentation shapes must match. "
            f"Got {input_zarr.shape} and {segmentation_zarr.shape}"
        )

    # Get metadata
    input_shape = input_zarr.shape
    dtype = input_zarr.dtype
    zarr_chunks = input_zarr.chunks

    # Create output zarr array
    output_zarr = zarr.open(
        str(output_path),
        mode="w",
        shape=input_shape,
        chunks=zarr_chunks,
        dtype=dtype,
    )

    # Calculate chunk grid
    n_chunks = tuple(int(np.ceil(input_shape[i] / chunk_shape[i])) for i in range(3))
    total_chunks = n_chunks[0] * n_chunks[1] * n_chunks[2]

    # Use repair_radius as border size
    border_size = (
        int(np.ceil(repair_radius)) + 2,
        int(np.ceil(repair_radius)) + 2,
        int(np.ceil(repair_radius)) + 2,
    )

    print(
        f"Processing {total_chunks} chunks of size {chunk_shape} "
        f"with border size {border_size}"
    )

    # Process chunks serially
    with tqdm(total=total_chunks, desc="Repairing breaks") as pbar:
        for i in range(n_chunks[0]):
            for j in range(n_chunks[1]):
                for k in range(n_chunks[2]):
                    pbar.update(1)

                    # Calculate core chunk slice
                    core_start = (
                        i * chunk_shape[0],
                        j * chunk_shape[1],
                        k * chunk_shape[2],
                    )
                    core_end = (
                        min((i + 1) * chunk_shape[0], input_shape[0]),
                        min((j + 1) * chunk_shape[1], input_shape[1]),
                        min((k + 1) * chunk_shape[2], input_shape[2]),
                    )
                    core_slice = tuple(
                        slice(core_start[dim], core_end[dim]) for dim in range(3)
                    )

                    # Calculate expanded slice with boundary
                    expanded_slice, actual_border = calculate_expanded_slice(
                        core_slice, border_size, input_shape
                    )

                    # Process this chunk
                    repair_breaks_chunk(
                        skeleton=input_zarr,
                        output_skeleton=output_zarr,
                        segmentation=segmentation_zarr,
                        expanded_slice=expanded_slice,
                        actual_border=actual_border,
                        repair_radius=repair_radius,
                    )
