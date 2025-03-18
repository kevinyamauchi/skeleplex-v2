"""Script to generate and view the skeletonization of an example 3D bifurcating tree."""

import napari

from skeleplex.data.bifurcating_tree import apply_dilation_3d, generate_tree_3d
from skeleplex.skeleton.skeletonize import apply_skeletonization

tree = generate_tree_3d(
    shape=(200, 200, 200),
    nodes=3,
    length_ratio=0.8,
    left_angle=30,
    right_angle=30,
    branch_length=50,
)

segmented_tree = apply_dilation_3d(tree, dilation_radius=3)

# Apply skeletonization
skeletonized_tree = apply_skeletonization(segmented_tree)

# Open Napari Viewer
viewer = napari.Viewer()
viewer.dims.ndisplay = 3  # Enables 3D visualization
viewer.add_image(segmented_tree, name="Tree", colormap="gray")
viewer.add_image(skeletonized_tree, name="Skeletonized Tree", colormap="blue")


napari.run()
