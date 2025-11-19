import numpy as np  # noqa: D100

from skeleplex.graph.skeleton_graph import SkeletonGraph
from skeleplex.measurements.angles import run_all_angle_metrics
from skeleplex.measurements.graph_properties import get_all_graph_properties
from skeleplex.measurements.utils import graph_attributes_to_df

# load graph
graph_path = "../example_data/skeleton_graph.json"
skeleton_graph = SkeletonGraph.from_json_file(graph_path)

# make sure origin and voxel size are set
skeleton_graph.origin = 0
skeleton_graph.to_directed(origin=skeleton_graph.origin)
skeleton_graph.voxel_size_um = (1, 1, 1)

# add measurements to graph, there are more measurements available
skeleton_graph.graph, _ = get_all_graph_properties(
    skeleton_graph.graph,
    prefix = "example",
    origin = skeleton_graph.origin,
    approx =True
    )

skeleton_graph.graph, output = run_all_angle_metrics(
    skeleton_graph.graph,
    sample_positions=np.linspace(0,0.2,10),
    approx=True
    )

# save graph with measurements
skeleton_graph.to_json_file("../example_data/skeleton_graph_measured.json")
# Export measurements to csv
df = graph_attributes_to_df(skeleton_graph.graph)
df.to_csv("../example_data/skeleton_graph_measured.csv", index=False)
