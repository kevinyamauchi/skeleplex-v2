"""Construct the skeleton graph from a skeleton image using lazy computation."""

import itertools

import dask.array as da
import numpy as np
import zarr

from skeleplex.graph.image_to_graph_lazy import (
    compute_degrees,
    construct_dataframe,
    remove_isolated_voxels,
    skeleton_image_to_graph,
)
from skeleplex.graph.skeleton_graph import SkeletonGraph

if __name__ == "__main__":
    # set to True to launch the viewer after graph creation
    view_skeleton = True

    # path to load the skeleton from
    skeleton_path = "skeleton.zarr"

    # path to save the voxel uid image to
    voxel_uid_path = "voxel_uids.zarr"

    # path to save the output graph json
    output_json_path = "skeleton_graph.json"

    # image voxel size in microns (isotropic)
    voxel_size = 20.0

    # load the skeleton image
    final_skeleton = da.from_zarr(skeleton_path)

    # compute degrees image and remove isolated voxels
    # (lazy)
    degrees_image = compute_degrees(final_skeleton)
    filtered_skeleton = remove_isolated_voxels(
        skeleton_image=final_skeleton,
        degrees_image=degrees_image,
    )

    save_here = zarr.open(
        voxel_uid_path,
        mode="w",
        shape=filtered_skeleton.shape,
        chunks=filtered_skeleton.chunks,
        dtype=np.int64,
    )
    first_unused_label = 1
    for inds in itertools.product(*map(range, filtered_skeleton.blocks.shape)):
        chunk = filtered_skeleton.blocks[inds]
        chunk_mem = chunk.compute()
        non_zero_coords = np.where(chunk_mem)
        chunk_mem = chunk_mem.astype(np.int64)

        unique_labels = np.arange(
            first_unused_label,
            first_unused_label + non_zero_coords[0].size,
            dtype=np.int64,
        )

        if non_zero_coords[0].size != 0:
            chunk_mem[tuple(non_zero_coords)] = unique_labels

        first_unused_label += non_zero_coords[0].size

        region = tuple(
            slice(
                inds[dim] * filtered_skeleton.chunks[dim][0],
                inds[dim] * filtered_skeleton.chunks[dim][0] + chunk_mem.shape[dim],
            )
            for dim in range(filtered_skeleton.ndim)
        )

        # save as zarr
        save_here[region] = chunk_mem

    binary_skeleton_uid = da.from_zarr(voxel_uid_path)

    edges_df = construct_dataframe(labeled_skeleton_image=binary_skeleton_uid)
    edges_df = edges_df.compute()

    # construct the graph
    nx_graph = skeleton_image_to_graph(
        skeleton_image=binary_skeleton_uid,
        degrees_image=degrees_image,
        graph_edges_df=edges_df,
        image_voxel_size_um=voxel_size,
    )
    skeleton_graph = SkeletonGraph.from_graph(
        graph=nx_graph,
        edge_coordinate_key="path",
        node_coordinate_key="node_coordinate",
        voxel_size_um=voxel_size,
    )

    # write the graph
    skeleton_graph.to_json_file(output_json_path)

    # import napari
    # viewer = napari.Viewer()
    # viewer.add_labels(filtered_skeleton.compute(), name="skeleton")
    # viewer.add_labels(binary_skeleton_uid.compute(), name="voxel_uids")
    # napari.run()

    if view_skeleton:
        from skeleplex.app import run, view_skeleton

        view_skeleton(graph_path=output_json_path, launch_widgets=False)
        run()
