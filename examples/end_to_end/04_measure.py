"""Make the measurements of branch length and diameters on a skeleton graph."""

from skeleplex.graph import SkeletonGraph
from skeleplex.graph.utils import write_slices_to_h5
from skeleplex.measurements.branches import (
    add_measurements_from_h5_to_graph,
    filter_and_segment_lumen,
)
from skeleplex.measurements.graph_properties import compute_level, get_parent_edges
from skeleplex.measurements.utils import graph_attributes_to_df

if __name__ == "__main__":
    # the ID of the node at the top of the tree
    origin_node_id = 1

    # path to the segmentation volume
    segmentation_path = "bifurcating_tree.zarr"

    # path to the skeleton graph json
    skeleton_graph_path = "skeleton_graph.json"

    # path to save the extracted slices
    slice_path = "branch_slices"

    # path to save the filtered slices
    filtered_slices_path = "filtered_branch_slices"

    # path to save the annotated skeleton graph
    annotated_graph_path = "skeleton_graph_annotated.json"

    # path to save the data table
    measurement_table_path = "skeleton_graph_measurements.csv"

    # load the graph
    skeleton_graph = SkeletonGraph.from_json_file(skeleton_graph_path)

    # make the graph directed
    skeleton_graph.to_directed(origin=origin_node_id)

    # annotate the graph with the generation (level) of each node and edge
    skeleton_graph.graph = compute_level(skeleton_graph.graph, origin=origin_node_id)

    # orient the direction of the edge splines
    skeleton_graph.orient_splines(approximate_positions=False)

    # annotate the graph with the length of each branch
    edge_lengths = skeleton_graph.compute_branch_lengths()

    # annotate the graph edges with the parent edge IDSs
    skeleton_graph.graph = get_parent_edges(skeleton_graph.graph)

    # annotate the edges with the parent edge start and end nodes
    for u, v in skeleton_graph.graph.edges():
        edge_data = skeleton_graph.graph[u][v]

        parent_edge = edge_data.get("parent_edge", [])
        if parent_edge:
            parent_edge_start = parent_edge[0][0]
            parent_edge_end = parent_edge[0][1]
        else:
            parent_edge_start = None
            parent_edge_end = None

        skeleton_graph.graph[u][v]["edge_start"] = u
        skeleton_graph.graph[u][v]["edge_end"] = v
        skeleton_graph.graph[u][v]["parent_edge_start"] = parent_edge_start
        skeleton_graph.graph[u][v]["parent_edge_end"] = parent_edge_end

    skeleton_graph.to_json_file(annotated_graph_path)

    # measure the branch diameters
    slices_dict = skeleton_graph.sample_volume_slices_from_spline_parallel(
        volume_path=segmentation_path,
        slice_spacing=0.2,
        slice_size_um=260,
        sample_grid_spacing_um=1,
        interpolation_order=0,
        max_generation=5,
    )

    write_slices_to_h5(
        file_path=slice_path,
        file_base="branch_",
        image_slices=slices_dict,
        segmentation_slices=None,
    )

    filter_and_segment_lumen(
        data_path=slice_path,
        save_path=filtered_slices_path,
        sam_checkpoint_path=None,
        resnet_predictor=None,
        eccentricity_thresh=0.85,
        circularity_thresh=0.5,
        find_lumen=False,
        segmentation_key="image",
    )

    skeleton_graph_with_measurements = add_measurements_from_h5_to_graph(
        graph_path=annotated_graph_path,
        input_path=filtered_slices_path,
    )

    # make the dataframe
    df = graph_attributes_to_df(skeleton_graph_with_measurements.graph)
    print(df[["lumen_diameter", "total_area"]])
