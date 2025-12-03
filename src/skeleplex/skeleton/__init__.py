"""Tools to create a skeleton image of a structure."""

from skeleplex.skeleton._break_detection import find_break_repairs, repair_breaks
from skeleplex.skeleton._break_detection_lazy import repair_breaks_lazy
from skeleplex.skeleton._chunked_label import (
    label_chunks_parallel,
    merge_touching_labels,
    relabel_parallel,
)
from skeleplex.skeleton._chunked_upscale import upscale_skeleton_parallel
from skeleplex.skeleton._segment import segment
from skeleplex.skeleton._skeletonize import skeletonize
from skeleplex.skeleton._upscale import upscale_skeleton
from skeleplex.skeleton._utils import get_skeletonization_model

__all__ = [
    "get_skeletonization_model",
    "segment",
    "skeletonize",
    "label_chunks_parallel",
    "find_break_repairs",
    "repair_breaks",
    "repair_breaks_lazy",
    "relabel_parallel",
    "merge_touching_labels",
    "upscale_skeleton",
    "upscale_skeleton_parallel",
]
