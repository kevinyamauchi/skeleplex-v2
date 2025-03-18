"""Generate and view the Dask-based skeletonization of a bifurcating tree."""

import dask.array as da
import napari

from skeleplex.data.bifurcating_tree import apply_dilation_3d, generate_tree_3d
from skeleplex.skeleton.skeletonize_efficient import apply_skeletenization_efficient

tree = generate_tree_3d(
    shape=(200, 200, 200),
    nodes=3,
    length_ratio=0.8,
    left_angle=30,
    right_angle=30,
    branch_length=50,
)

segmented_tree = apply_dilation_3d(tree, dilation_radius=3)

dask_tree = da.from_array(segmented_tree)

# Apply Dask-based skeletonization
skeletonized_tree_dask = apply_skeletenization_efficient(dask_tree, depth=3)

# Open Napari Viewer
viewer = napari.Viewer()
viewer.dims.ndisplay = 3  # Enables 3D visualization
viewer.add_image(segmented_tree, name="Segmented Image (Dilated)", colormap="gray")
viewer.add_image(
    skeletonized_tree_dask, name="Dask Skeletonized Image", colormap="red", opacity=0.7
)


napari.run()
