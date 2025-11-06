"""Create a simple tree data as zarr file."""

import dask.array as da

from skeleplex.data.bifurcating_tree import apply_dilation_3d, generate_tree_3d

# set to True to visualize the generated tree
visualize = True

# path to save the zarr file
zarr_path = "bifurcating_tree.zarr"

# size of the chunks (ZYX)
chunk_size = (25, 25, 25)

# Generate tree
tree = generate_tree_3d(
    shape=(50, 100, 100),
    num_bifurcations=2,
    branch_length=40,
    z_layer=25,
    left_angle=60,
    right_angle=60,
)

# dilate
dilated_tree = apply_dilation_3d(tree, dilation_radius=4)

# save as zarr

dask_array = da.from_array(dilated_tree, chunks=chunk_size)
dask_array.to_zarr(zarr_path, overwrite=True)
print(dilated_tree.shape)
if visualize:
    import napari

    viewer = napari.Viewer()
    viewer.add_image(dilated_tree, name="dilated_tree")
    napari.run()
