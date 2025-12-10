"""Utilities for operating on the skeleton."""

from skeleplex.utils._chunked import (
    calculate_expanded_slice,
    get_boundary_slices,
    iteratively_process_chunks_3d,
)
from skeleplex.utils._geometry import line_segments_in_aabb, points_in_aabb
from skeleplex.utils._inference_slurm import (
    build_sbatch_command,
    infer_on_chunk,
    initialize_parallel_inference,
)
from skeleplex.utils._tta import TTAWrapper
from skeleplex.utils._tta_augmentations import (
    FlipX,
    FlipY,
    FlipZ,
    Identity,
    Rot90YX,
    Rot90ZX,
    Rot90ZY,
    Rot180YX,
    Rot180ZX,
    Rot180ZY,
    Rot270YX,
    Rot270ZX,
    Rot270ZY,
)

__all__ = [
    "iteratively_process_chunks_3d",
    "line_segments_in_aabb",
    "points_in_aabb",
    "get_boundary_slices",
    "calculate_expanded_slice",
    "initialize_parallel_inference",
    "infer_on_chunk",
    "build_sbatch_command",
    "Identity",
    "FlipZ",
    "FlipY",
    "FlipX",
    "Rot90ZY",
    "Rot180ZY",
    "Rot270ZY",
    "Rot90ZX",
    "Rot180ZX",
    "Rot270ZX",
    "Rot90YX",
    "Rot180YX",
    "Rot270YX",
    "TTAWrapper",
]
