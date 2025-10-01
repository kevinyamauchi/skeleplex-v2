# Tutorial: Lazy Graph Construction from 3D Segmentation

This tutorial demonstrates how to compute a distance transform, perform chunkwise skeleton prediction using a pretrained model, and construct a skeleton graph from a 3D segmentation using the SkelePlex framework.

## Step 1: Generate Example Data

We start with a synthetic 3D bifurcating tree structure. This serves as an example of a segmented object with branching topology.
All scripts used in this rundown are found in [`examples/segmentation_to_graph/`](../examples/segmentation_to_graph/) .
You can generate the data using the provided script 'create_example_data.py', which:

- Creates a 3D binary tree-like structure
- Applies 3D dilation to thicken the branches
- Saves the result in zarr format as 'bifurcating_tree.zarr'

This file will be used as the input segmentation for the subsequent processing steps.


## Step 2: Predict Skeleton from Segmentation

The script 'predict_skeleton.py' computes a local normalized distance transform from the binary segmentation and performs skeleton prediction using a pretrained model.

This step includes:

- Computing the local distance field using chunked processing
- Predicting a skeleton using the pretrained 'MultiscaleSkeletonizationNet'
- Masking the predicted skeleton with the original segmentation
- Saving the outputs in zarr format

The resulting skeleton will serve as the basis for graph construction in the next step.


## Step 3: Postprocess and Threshold the Skeleton Prediction

The script 'create_skeleton.py' converts the skeleton prediction into a binary skeleton mask and refines it using classical morphological skeletonization.

This step includes:

- Thresholding the predicted skeleton to create a binary mask
- Applying a voxel-wise skeletonization ('skimage.morphology.skeletonize') in chunks using 'map_overlap'
- Saving the final skeleton to zarr format

The output is a binarized skeleton that can be used for graph construction.


## Step 4: Construct the Graph from the Skeleton

The script 'create_graph.py' converts the final binary skeleton into a graph representation and stores it as a JSON file for further analysis or visualization.

This step includes:

- Computing the degree of each skeleton voxel
- Removing isolated voxels with no connected neighbors
- Assigning unique IDs to each voxel in the skeleton
- Constructing edge paths and a connectivity table from the labeled components
- Creating a 'SkeletonGraph' object from the voxel graph
- Saving the resulting graph in JSON format

The output graph structure captures the connectivity and geometry of the original 3D skeleton and can be further processed or visualized using the SkelePlex viewer or other tools.
