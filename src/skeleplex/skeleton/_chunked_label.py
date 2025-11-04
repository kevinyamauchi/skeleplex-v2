from functools import partial
from itertools import product
from multiprocessing import Lock, Value, get_context
from typing import Literal

import numpy as np
import zarr
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import label

# Global variables to hold shared state in worker processes
_offset_counter = None
_counter_lock = None


def _init_worker(offset_counter, counter_lock):
    """Initialize worker process with shared state."""
    global _offset_counter, _counter_lock
    _offset_counter = offset_counter
    _counter_lock = counter_lock


def create_chunk_slices(
    array_shape: tuple[int, ...], chunk_shape: tuple[int, ...]
) -> list[tuple[slice, ...]]:
    """
    Create a list of slice tuples for iterating over an array in chunks.

    Parameters
    ----------
    array_shape : tuple of int
        Shape of the array to be chunked (e.g., (1024, 2048, 2048))
    chunk_shape : tuple of int
        Shape of each chunk (e.g., (256, 512, 512))

    Returns
    -------
    list of tuple of slice
        list where each element is a tuple of slices for one chunk.
        The tuple has the same length as array_shape.
    """
    if len(array_shape) != len(chunk_shape):
        raise ValueError("array_shape and chunk_shape must have same length")

    # Calculate number of chunks along each dimension
    n_chunks_per_dim = [
        (size + chunk_size - 1) // chunk_size  # Ceiling division
        for size, chunk_size in zip(array_shape, chunk_shape, strict=False)
    ]

    # Generate all chunk indices
    chunk_slices = []
    for chunk_indices in product(*[range(n) for n in n_chunks_per_dim]):
        slices = tuple(
            slice(idx * chunk_size, min((idx + 1) * chunk_size, array_size))
            for idx, chunk_size, array_size in zip(
                chunk_indices, chunk_shape, array_shape, strict=False
            )
        )
        chunk_slices.append(slices)

    return chunk_slices


def _label_chunk_with_offset(
    chunk_slices: tuple[slice, ...], input_path: str, output_path: str
) -> tuple[tuple[slice, ...], int]:
    """
    Process a single chunk: label connected components and apply offset.

    Uses global variables _offset_counter and _counter_lock set by initializer.
    This should not be used independently; use label_chunks_parallel instead.

    Parameters
    ----------
    chunk_slices : tuple of slice
        Slice tuple defining the chunk location in the array
    input_path : str
        Path to input zarr array
    output_path : str
        Path to output zarr array

    Returns
    -------
    tuple
        (chunk_slices, max_label) for logging/debugging
    """
    global _offset_counter, _counter_lock

    # Load input chunk
    input_zarr = zarr.open(input_path, mode="r")
    chunk_data = input_zarr[chunk_slices]

    # Label connected components (returns 0 for background)
    labeled_chunk = label(chunk_data)
    max_label = int(labeled_chunk.max())

    # Get offset atomically and update counter
    with _counter_lock:
        my_offset = _offset_counter.value
        _offset_counter.value += max_label

    # Apply offset to non-background pixels
    if max_label > 0:
        mask = labeled_chunk > 0
        labeled_chunk[mask] += my_offset

    # Write to output zarr
    output_zarr = zarr.open(output_path, mode="r+")
    output_zarr[chunk_slices] = labeled_chunk

    return (chunk_slices, max_label)


def label_chunks_parallel(
    input_path: str,
    output_path: str,
    chunk_shape: tuple[int, ...],
    n_processes: int = 4,
    pool_type: Literal["spawn", "fork", "forkserver", "thread"] = "fork",
) -> int:
    """
    Label connected components in a large zarr image using parallel processing.

    Parameters
    ----------
    input_path : str
        Path to input zarr array
    output_path : str
        Path to output zarr array (will be created if doesn't exist)
    chunk_shape : tuple of int
        Shape of chunks to process in parallel. This will be the chunk shape
        of the output array.
    n_processes : int, default=4
        Number of parallel processes/threads
    pool_type : {'spawn', 'fork', 'forkserver', 'thread'}, default='spawn'
        Type of multiprocessing context to use.
        - 'spawn': Start fresh Python process (safest, works on all platforms)
        - 'fork': Copy parent process (faster but can have issues with threads)
        - 'forkserver': Hybrid approach (Unix only)
        - 'thread': Use threading instead of multiprocessing (good for I/O bound)

    Returns
    -------
    int
        Total number of unique labels assigned

    Notes
    -----
    - Input zarr must already exist
    - Output zarr will be created with same shape/dtype as input if it doesn't exist
    - Components spanning chunk boundaries will receive different labels
    """
    # Open input zarr to get metadata
    input_zarr = zarr.open(input_path, mode="r")
    array_shape = input_zarr.shape

    # Create the output zarr
    _ = zarr.create_array(
        output_path, shape=array_shape, chunks=chunk_shape, dtype=np.uint64
    )

    # Create list of chunk slices
    chunk_slices_list = create_chunk_slices(array_shape, chunk_shape)

    print(
        f"Processing {len(chunk_slices_list)} chunks using {n_processes} "
        f"{pool_type} workers"
    )

    # Process chunks in parallel
    if pool_type == "thread":
        from multiprocessing.pool import ThreadPool

        offset_counter = Value("i", 0)
        counter_lock = Lock()
        pool = ThreadPool(
            n_processes,
            initializer=_init_worker,
            initargs=(offset_counter, counter_lock),
        )
    else:
        ctx = get_context(pool_type)
        offset_counter = ctx.Value("i", 0)
        counter_lock = ctx.Lock()
        pool = ctx.Pool(
            n_processes,
            initializer=_init_worker,
            initargs=(offset_counter, counter_lock),
        )

    # Create the processing function
    process_func = partial(
        _label_chunk_with_offset, input_path=input_path, output_path=output_path
    )

    try:
        _ = pool.map(process_func, chunk_slices_list)
    finally:
        pool.close()
        pool.join()

    total_labels = offset_counter.value

    return total_labels


def _find_touching_labels(
    region_slice: tuple[slice, slice, slice], label_image_path: str
) -> np.ndarray:
    """
    Find pairs of labels that are touching within a region using 26-connectivity.

    Parameters
    ----------
    region_slice: tuple[slice, slice, slice]
        slice objects defining the region to check
    label_image_path: str
        path to zarr array containing labeled image

    Returns
    -------
    np.ndarray
        (n_pairs, 2) array where each row is [label_A, label_B]
        with label_A < label_B. Returns empty (0, 2) array if no touching pairs.
    """
    # Load the region from zarr
    label_image = zarr.open(str(label_image_path), mode="r")
    region = label_image[region_slice]

    # Get unique labels in the region (exclude 0 which is background)
    unique_labels = np.unique(region)
    unique_labels = unique_labels[unique_labels != 0]

    # Create 26-connectivity structure (3D, connectivity=3 includes all diagonals)
    connectivity_structure = generate_binary_structure(3, 3)

    # Set to store unique touching pairs
    touching_pairs = set()

    # For each label, find what it touches
    for label_A in unique_labels:
        # Create binary mask for this label
        binary_mask = region == label_A

        # Dilate by 1 voxel in all 26 directions
        dilated_mask = binary_dilation(binary_mask, structure=connectivity_structure)

        # Find labels in the dilated region
        touching_region = region[dilated_mask]
        touching_labels = np.unique(touching_region)

        # Exclude the label itself and background (0)
        touching_labels = touching_labels[
            (touching_labels != label_A) & (touching_labels != 0)
        ]

        # Add pairs (always store as min, max to avoid duplicates)
        for label_B in touching_labels:
            pair = (min(label_A, label_B), max(label_A, label_B))
            touching_pairs.add(pair)

    # Convert to numpy array
    if len(touching_pairs) == 0:
        return np.empty((0, 2), dtype=region.dtype)

    result = np.array(list(touching_pairs), dtype=region.dtype)
    return result


def _make_label_mapping(
    touching_pairs: np.ndarray, max_label_value: int
) -> dict[int, int]:
    """
    Create a label mapping based on connected components of touching labels.

    Labels in the same connected component are mapped to the maximum label
    value within that component.

    Parameters
    ----------
    touching_pairs : np.ndarray
        Array of shape (n_pairs, 2) with pairs of touching labels.
    max_label_value : int
        Maximum label value in the entire image.

    Returns
    -------
    dict[int, int]
        Dictionary mapping original labels to new labels. Only includes labels
        that need to change (excludes identity mappings like {5: 5}).
    """
    # Handle empty touching_pairs
    if len(touching_pairs) == 0:
        return {}

    # Build adjacency matrix
    # Matrix size is (max_label_value + 1) to accommodate labels
    # from 0 to max_label_value
    n_labels = max_label_value + 1

    # Extract row and column indices from touching_pairs
    rows = touching_pairs[:, 0]
    cols = touching_pairs[:, 1]

    # Create data for both directions (undirected graph)
    # Add both [a, b] and [b, a] edges
    row_indices = np.concatenate([rows, cols])
    col_indices = np.concatenate([cols, rows])
    data = np.ones(len(row_indices), dtype=np.uint8)

    # Build sparse adjacency matrix
    adjacency_matrix = csr_matrix(
        (data, (row_indices, col_indices)), shape=(n_labels, n_labels)
    )

    # Find connected components
    n_components, component_labels = connected_components(
        adjacency_matrix, directed=False, return_labels=True
    )

    # Get unique labels that appear in touching_pairs
    unique_labels = np.unique(touching_pairs)

    # Find max label in each component
    # component_max[component_id] = max_label in that component
    component_max = {}

    for label_value in unique_labels:
        component_id = component_labels[label_value]

        if component_id not in component_max:
            component_max[component_id] = label_value
        else:
            component_max[component_id] = max(component_max[component_id], label_value)

    # Create mapping: only include labels that change
    label_mapping = {}

    for label_value in unique_labels:
        component_id = component_labels[label_value]
        max_label_in_component = component_max[component_id]

        # Only add to mapping if label changes
        if label_value != max_label_in_component:
            label_mapping[label_value] = max_label_in_component

    return label_mapping
