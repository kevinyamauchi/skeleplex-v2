from skeleplex.measurements.branches import (  # noqa: D100
    add_measurements_from_h5_to_graph,
    filter_and_segment_lumen,
)
from skeleplex.measurements.lumen_classifier import ResNet3ClassPredictor

# This script requires a SAM2 model to segment the lumen
# Download from https://github.com/facebookresearch/sam2/tree/main
slice_path = "../example_data/branch_slices.h5"
graph_path = "../example_data/skeleton_graph.json"
filtered_slice_path = "../example_data/filtered_branch_slices.h5"

sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
resnet_checkpoint = (
    "resnet_classification/checkpoints/resnet3class-epoch=22-val_loss=0.11.ckpt"
)

# Requires the training of your own ResNet classifier.
# See examples/measurements/train_resnet_classifier.py
resnet_class_predictor = ResNet3ClassPredictor(
    checkpoint_path=resnet_checkpoint, device="cuda:0", num_classes=3
)

filter_and_segment_lumen(
    data_path=slice_path,
    save_path=filtered_slice_path,
    sam_checkpoint_path=sam2_checkpoint,
    resnet_predictor=resnet_class_predictor,
    eccentricity_thresh=0.85,
    circularity_thresh=0.5,
    find_lumen=False,  # Set to true to actually use the lumen finder
)

# Now add measurements to graph
add_measurements_from_h5_to_graph(
    graph_path=graph_path,
    input_path=filtered_slice_path,
)
