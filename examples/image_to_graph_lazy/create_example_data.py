"""Example script to generate and save a 3D bifurcating tree as zarr file."""

import dask.array as da

from skeleplex.data.bifurcating_tree import apply_dilation_3d, generate_tree_3d

# Generate tree
tree = generate_tree_3d(
        shape=(5, 100, 100),
        num_bifurcations=2,
        branch_length=40,
        z_layer=2,
        left_angle=60,
        right_angle=60,
    )

# dilate
dilated_tree = apply_dilation_3d(tree, dilation_radius=2)

# save as zarr
zarr_path = "examples/image_to_graph_lazy/bifurcating_tree.zarr"
dask_array = da.from_array(dilated_tree)
dask_array.to_zarr(zarr_path, overwrite=True)
