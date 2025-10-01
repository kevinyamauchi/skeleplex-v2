import dask.array as da  # noqa: D100
from dask_image.ndfilters import gaussian_filter

from skeleplex.graph.skeleton_graph import SkeletonGraph
from skeleplex.graph.utils import write_slices_to_h5


def main():  # noqa: D103
    # Define paths
    volume_path = "../example_data/blurred_volume.zarr"
    segmentation_path = "../example_data/bifurcating_tree.zarr"
    graph_path = "../example_data/skeleton_graph.json"
    slice_path = "../example_data/branch_slices.h5"
    # Load skeleton graph
    skeleton_graph = SkeletonGraph.from_json_file(graph_path)
    # We need to transform the graph to a directed graph
    # For that we need to define a root node
    # Here we simply take the first node
    skeleton_graph.origin = 0
    skeleton_graph.to_directed(origin=skeleton_graph.origin)
    # We need to define a voxel size in um for the measurements
    # Set voxel size in um
    skeleton_graph.voxel_size_um = (1, 1, 1)

    # create mock image
    segmentation = da.from_zarr(segmentation_path)
    volume = segmentation.astype(float)
    # blur image to simulate a real microscopy image
    volume = gaussian_filter(volume, sigma=1)
    volume.to_zarr(volume_path, overwrite=True)

    # Extract orthogonal slices for each branch
    image_slice_dict, seg_slice_dict = (
        skeleton_graph.sample_volume_slices_from_spline_parallel(
            volume_path=volume_path,
            slice_spacing=0.1,
            slice_size=20,
            segmentation_path=segmentation_path,
        )
    )
    write_slices_to_h5(
        file_path=slice_path,
        file_base="branch_",
        image_slices=image_slice_dict,
        segmentation_slices=seg_slice_dict,
    )


if __name__ == "__main__":
    main()
