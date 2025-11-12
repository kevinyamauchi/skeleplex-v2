import numpy as np  # noqa: D100

from skeleplex.graph.skeleton_graph import SkeletonGraph
from skeleplex.measurements.angles import compute_midline_branch_angle_spline
from skeleplex.measurements.graph_properties import (
    compute_branch_length,
    compute_level,
    compute_number_of_tips_connected_to_edges,
    get_daughter_edges,
    get_sister_edges,
)
from skeleplex.measurements.utils import graph_attributes_to_df

# load graph
graph_path = "../example_data/skeleton_graph.json"
skeleton_graph = SkeletonGraph.from_json_file(graph_path)

# make sure origin and voxel size are set
skeleton_graph.origin = 0
skeleton_graph.to_directed(origin=skeleton_graph.origin)
skeleton_graph.voxel_size_um = (1, 1, 1)

# add measurements to graph, there are more measurements available
skeleton_graph.graph = get_sister_edges(skeleton_graph.graph)
skeleton_graph.graph = get_daughter_edges(skeleton_graph.graph)
skeleton_graph.graph = compute_level(skeleton_graph.graph, origin=0)
skeleton_graph.graph = compute_branch_length(skeleton_graph.graph)
skeleton_graph.graph = compute_number_of_tips_connected_to_edges(skeleton_graph.graph)
skeleton_graph.graph, _ = compute_midline_branch_angle_spline(
    skeleton_graph.graph, sample_positions=np.linspace(0, 0.3, 10)
)


# save graph with measurements
skeleton_graph.to_json_file("../example_data/skeleton_graph_measured.json")
# Export measurements to csv
df = graph_attributes_to_df(skeleton_graph.graph)
df.to_csv("../example_data/skeleton_graph_measured.csv", index=False)
