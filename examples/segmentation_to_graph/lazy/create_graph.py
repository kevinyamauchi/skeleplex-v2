"""Example script to convert a skeleton zarr image into a SkeletonGraph."""

import dask.array as da

from skeleplex.graph.image_to_graph_lazy import (
    assign_unique_ids,
    compute_degrees,
    construct_dataframe,
    remove_isolated_voxels,
    skeleton_image_to_graph,
)
from skeleplex.graph.skeleton_graph import SkeletonGraph

path_to_skeleton = "../../example_data/skeleton.zarr"

skeleton_image = da.from_zarr(path_to_skeleton)

degrees_image = compute_degrees(skeleton_image)

filtered_skeleton = remove_isolated_voxels(
    skeleton_image=skeleton_image,
    degrees_image=degrees_image,
)

labeled_skeleton, num_labels = assign_unique_ids(filtered_skeleton)

edges_df = construct_dataframe(labeled_skeleton_image=labeled_skeleton)

edges_df = edges_df.compute()

nx_graph = skeleton_image_to_graph(
    skeleton_image=filtered_skeleton,
    degrees_image=degrees_image,
    graph_edges_df=edges_df,
)


skeleton_graph = SkeletonGraph.from_graph(
    graph=nx_graph, edge_coordinate_key="path", node_coordinate_key="node_coordinate"
)


output_json_path = "../../example_data/skeleton_graph.json"
skeleton_graph.to_json_file(output_json_path)
