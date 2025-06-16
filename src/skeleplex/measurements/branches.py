import logging  # noqa
import os
import networkx as nx
import h5py
import numpy as np
import skimage as ski
import torch
from tqdm import tqdm
import concurrent.futures


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from skeleplex.measurements.utils import grey2rgb, radius_from_area
from skeleplex.graph.skeleton_graph import SkeletonGraph

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def filter_and_segment_lumen(
    data_path,
    save_path,
    sam_checkpoint_path,
    resnet_predictor,
    eccentricity_thresh=0.7,
    circularity_thresh=0.5,
    find_lumen=True,
):
    """
    Filter and segment the lumen in the image slices.

    Uses the spline to seed an prompt for SAM2
    https://github.com/facebookresearch/sam2/tree/main

    And a resnet classifier to classify the slices into lumen, branches and bad.

    Parameters
    ----------
    data_path : str
        Path to the input data directory containing .h5 files.
    save_path : str
        Path to the output directory where filtered .h5 files will be saved.
    sam_checkpoint_path : str
        Path to the SAM2 checkpoint file.
    resnet_predictor : ResNet3ClassClassifier
        ResNet classifier for predicting classes.
    eccentricity_thresh : float
        Eccentricity threshold for filtering slices.
    circularity_thresh : float
        Circularity threshold for filtering slices.
    find_lumen : bool
        Whether to find the lumen using SAM2 or just to filter for
        eccentricity and circularity.


    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logger.info(f"Created directory: {save_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # viewer = napari.Viewer()
    sam2_checkpoint = sam_checkpoint_path
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files = [f for f in os.listdir(data_path) if f.endswith(".h5")]
    # only do those that are npot in the save_path
    files = [f for f in files if not os.path.exists(os.path.join(save_path, f))]
    logger.info(f"Found {len(files)} files to process.")
    files = tqdm(files, desc="Processing files")
    for file in files:
        logger.info(f"Processing {file}")

        with h5py.File(os.path.join(data_path, file), "r") as f:
            image_slices = f["image"][:]
            segmentation_slices = f["segmentation"][:] != 0

        label_slices_filt = np.zeros_like(segmentation_slices, dtype=np.uint8)
        index_to_remove = []

        for i in range(len(image_slices)):
            image_slice = image_slices[i]
            segmentation_slice = segmentation_slices[i]
            label_slice = ski.measure.label(segmentation_slice)
            h, w = image_slice.shape
            central_label = label_slice[h // 2, w // 2]

            # Skip if no central label
            if central_label == 0:
                index_to_remove.append(i)
                continue

            # Remove if touching the border
            if (
                np.any(label_slice[0, :] == central_label)
                or np.any(label_slice[-1, :] == central_label)
                or np.any(label_slice[:, 0] == central_label)
                or np.any(label_slice[:, -1] == central_label)
            ):
                index_to_remove.append(i)
                continue

            label_slice[label_slice != central_label] = 0
            label_slice[label_slice == central_label] = 1

            # Check eccentricity
            props = ski.measure.regionprops(label_slice)
            if props[0].eccentricity > eccentricity_thresh:
                index_to_remove.append(i)
                continue

            # Check circularity
            circularity = 4 * np.pi * props[0].area / (props[0].perimeter ** 2)
            if circularity < circularity_thresh:
                index_to_remove.append(i)
                continue

            if find_lumen:
                # Segment using SAM2
                image_slice_rgb = grey2rgb(image_slice)
                predictor.set_image(image_slice_rgb)
                sam_point = np.array([[h // 2, w // 2]])
                sam_label = np.array([1])
                sam_mask, _, _ = predictor.predict(
                    point_coords=sam_point,
                    point_labels=sam_label,
                    multimask_output=True,
                )

                mask_img1 = image_slice * sam_mask[0]
                mask_img2 = image_slice * sam_mask[1]
                mask_img3 = image_slice * sam_mask[2]

                # crop to bouynding box
                min_x, min_y, max_x, max_y = ski.measure.regionprops(
                    sam_mask[0].astype(int)
                )[0].bbox
                mask_img1 = mask_img1[min_x:max_x, min_y:max_y]

                min_x, min_y, max_x, max_y = ski.measure.regionprops(
                    sam_mask[1].astype(int)
                )[0].bbox
                mask_img2 = mask_img2[min_x:max_x, min_y:max_y]

                min_x, min_y, max_x, max_y = ski.measure.regionprops(
                    sam_mask[2].astype(int)
                )[0].bbox
                mask_img3 = mask_img3[min_x:max_x, min_y:max_y]

                preds = []
                for j, mask_img in enumerate([mask_img1, mask_img2, mask_img3]):
                    pred_class, conf = resnet_predictor.predict(mask_img)
                    preds.append({"index": j, "class": pred_class, "conf": conf})

                # Initialize label mask
                label_with_lumen = np.zeros_like(label_slice, dtype=np.uint8)
                logger.info([p["class"] for p in preds])

                # Handle lumen (class 0)
                lumen_preds = [p for p in preds if p["class"] == 0]
                if lumen_preds:
                    logger.info("Found lumen")
                    best_lumen = max(lumen_preds, key=lambda x: x["conf"])
                    label_with_lumen[sam_mask[best_lumen["index"]] == 1] = 2

                # Handle branches (class 1)
                class1_preds = [p for p in preds if p["class"] == 1]
                if class1_preds:
                    best_class1 = max(class1_preds, key=lambda x: x["conf"])
                    label_with_lumen[
                        (sam_mask[best_class1["index"]] == 1) & (label_with_lumen == 0)
                    ] = 1
                else:
                    # If no class 1 was found, but something is labeled in the
                    # original slice, fill in as class 1
                    label_with_lumen[(label_slice != 0) & (label_with_lumen == 0)] = 1

                # Handle full match for class 2 (e.g., all bad)
                if all(p["class"] == 2 for p in preds):
                    label_with_lumen = label_slice.copy()

                label_slice = label_with_lumen

            label_slices_filt[i] = label_slice

        # Remove invalid slices
        image_slice_filt = np.delete(image_slices, index_to_remove, axis=0)
        label_slices_filt = np.delete(label_slices_filt, index_to_remove, axis=0)

        with h5py.File(os.path.join(save_path, file), "w") as f:
            f.create_dataset("image", data=image_slice_filt)
            f.create_dataset("segmentation", data=label_slices_filt)


def filter_for_iterative_lumens(data_path, save_path):
    """Filter for iterative lumens across multiple files in parallel."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logger.info(f"Created directory: {save_path}")

    files = [f for f in os.listdir(data_path) if f.endswith(".h5")]
    logger.info(f"Found {len(files)} files to process.")
    files = tqdm(files, desc="Processing files")

    # Pack arguments for parallel processing
    file_args = [(f, data_path, save_path) for f in files]

    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        list(executor.map(filter_file_for_iterative_lumens, file_args))


def filter_file_for_iterative_lumens(args):
    """Filter a single HDF5 file for iterative lumens."""
    file, data_path, save_path = args
    logger.info(f"Processing {file}")

    input_file_path = os.path.join(data_path, file)
    output_file_path = os.path.join(save_path, file)
    try:
        with h5py.File(input_file_path, "r") as f:
            image_slices = f["image"][:]
            segmentation_slices = f["segmentation"][:]
    except Exception as e:
        logger.warning(f"Error loading {file}: {e}")
        return

    if np.sum(segmentation_slices == 2) == 0:
        logger.info(f"No label 2 in file {file}, skipping.")
        return

    index_to_remove = []
    for i, label_slice in enumerate(segmentation_slices):
        if i == 0 or i == len(segmentation_slices) - 1:
            continue

        if np.sum(label_slice == 2) == 0:
            if (
                np.sum(segmentation_slices[i - 1] == 2) > 0
                and np.sum(segmentation_slices[i + 1] == 2) > 0
            ):
                index_to_remove.append(i)

    if not index_to_remove:
        logger.info(f"No slices to remove in file {file}")
        return

    image_slice_filt = np.delete(image_slices, index_to_remove, axis=0)
    label_slices_filt = np.delete(segmentation_slices, index_to_remove, axis=0)

    with h5py.File(output_file_path, "w") as f:
        f.create_dataset("image", data=image_slice_filt)
        f.create_dataset("segmentation", data=label_slices_filt)
    logger.info(f"Filtered and saved {file}")


def fix_only_lumen(segmentation_slice):
    """Fix segmentation slices containing only lumen."""
    boundary_lumen = set(
        map(
            tuple,
            np.round(
                np.concatenate(ski.measure.find_contours(segmentation_slice == 2, 0.5))
            ).astype(np.int32),
        )
    )
    boundary_background = set(
        map(
            tuple,
            np.round(
                np.concatenate(ski.measure.find_contours(segmentation_slice == 0, 0.5))
            ).astype(np.int32),
        )
    )
    return bool(boundary_lumen & boundary_background)


def add_file_to_graph(file):
    """Process a single HDF5 file."""
    try:
        with h5py.File(file, "r") as f:
            image_slices = f["image"][:]
            segmentation_slices = f["segmentation"][:]
    except Exception as e:
        logger.warning(f"Error loading {file}: {e}")
        return None

    file_name = os.path.basename(file)
    start_node = int(file_name.split("_")[3])
    end_node = int(file_name.split("_")[5].split(".")[0])

    tissue_radius_branch = []
    lumen_radius_branch = []
    minor_axis_branch = []
    major_axis_branch = []
    total_area_branch = []

    for slice_index, (_, segmentation_slice) in enumerate(
        zip(image_slices, segmentation_slices, strict=False)
    ):
        if np.sum(segmentation_slice == 2) > 0:
            if np.sum(segmentation_slice == 1) == 0 and fix_only_lumen(
                segmentation_slice
            ):
                logger.info(f"Fixing {file}, slice {slice_index}")
                segmentation_slice[segmentation_slice == 2] = 1
                segmentation_slices[slice_index] = segmentation_slice

            label_slice = ski.measure.label((segmentation_slice != 0).astype(np.uint8))
            props = ski.measure.regionprops(label_slice)

            if props:
                minor_axis = props[0].minor_axis_length
                major_axis = props[0].major_axis_length
                total_area = props[0].area
                minor_axis_branch.append(minor_axis)
                major_axis_branch.append(major_axis)
                total_area_branch.append(total_area)

            if np.sum(segmentation_slice == 1) > 0:
                lumen_label = (segmentation_slice == 2).astype(np.uint8)
                tissue_label = (segmentation_slice == 1).astype(np.uint8)

                lumen_props = ski.measure.regionprops(ski.measure.label(lumen_label))
                tissue_props = ski.measure.regionprops(ski.measure.label(tissue_label))

                if lumen_props and tissue_props:
                    lumen_area = lumen_props[0].area
                    tissue_area = tissue_props[0].area
                    total_area = lumen_area + tissue_area
                    total_radius = radius_from_area(total_area)
                    lumen_radius = radius_from_area(lumen_area)
                    tissue_radius = total_radius - lumen_radius
                    tissue_radius_branch.append(tissue_radius)
                    lumen_radius_branch.append(lumen_radius)
            else:
                # no tissue label, full region is tissue
                tissue_radius_branch.append(minor_axis / 2)
                lumen_radius_branch.append(0)
        else:
            # completely closed (no lumen)
            label_slice = ski.measure.label((segmentation_slice != 0).astype(np.uint8))
            props = ski.measure.regionprops(label_slice)
            if props:
                minor_axis = props[0].minor_axis_length
                major_axis = props[0].major_axis_length
                total_area = props[0].area
                minor_axis_branch.append(minor_axis)
                major_axis_branch.append(major_axis)
                total_area_branch.append(total_area)
                tissue_radius_branch.append(minor_axis / 2)
                lumen_radius_branch.append(0)

    return (
        start_node,
        end_node,
        np.mean(np.array(lumen_radius_branch) * 2),
        np.std(np.array(lumen_radius_branch) * 2),
        np.mean(tissue_radius_branch),
        np.std(tissue_radius_branch),
        np.mean(total_area_branch),
        np.std(total_area_branch),
        np.mean(minor_axis_branch),
        np.std(minor_axis_branch),
        np.mean(major_axis_branch),
        np.std(major_axis_branch),
    )


def add_measurements_from_h5_to_graph(graph_path, input_path):
    """
    Add measurements from HDF5 files to the skeleton graph.

    The slice names need to be in the format:

    {base}_{name}_{start_node}_{end_node}.h5

    Parameters
    ----------
    graph_path : str
        Path to the skeleton graph JSON file.
    input_path : str
        Path to the directory containing HDF5 files with the segmented slices.

    Returns
    -------
    SkeletonGraph
        The updated skeleton graph with measurements added.
    """
    # Load skeleton graph
    skeleton_graph = SkeletonGraph.from_json_file(graph_path)

    # Prepare attributes
    attribute_names = [
        "lumen_diameter",
        "tissue_thickness",
        "total_area",
        "minor_axis",
        "major_axis",
    ]
    for attr in attribute_names + [f"{a}_sd" for a in attribute_names]:
        nx.set_edge_attributes(skeleton_graph.graph, {}, name=attr)

    measurement_dicts = {
        key: {}
        for key in (
            "lumen_diameter",
            "tissue_thickness",
            "total_area",
            "minor_axis",
            "major_axis",
            "lumen_diameter_sd",
            "tissue_thickness_sd",
            "total_area_sd",
            "minor_axis_sd",
            "major_axis_sd",
        )
    }

    # Get list of HDF5 files
    files = [f for f in os.listdir(input_path) if f.endswith(".h5")]
    # add input path to files
    files = [os.path.join(input_path, f) for f in files]

    # Process files in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        for result in tqdm(executor.map(add_file_to_graph, files), total=len(files)):
            if result:
                results.append(result)

    # Fill measurement dicts
    for (
        start_node,
        end_node,
        lumen_diameter_mean,
        lumen_diameter_sd,
        tissue_thickness_mean,
        tissue_thickness_sd,
        total_area_mean,
        total_area_sd,
        minor_axis_mean,
        minor_axis_sd,
        major_axis_mean,
        major_axis_sd,
    ) in results:
        edge = (start_node, end_node)
        measurement_dicts["lumen_diameter"][edge] = lumen_diameter_mean
        measurement_dicts["lumen_diameter_sd"][edge] = lumen_diameter_sd
        measurement_dicts["tissue_thickness"][edge] = tissue_thickness_mean
        measurement_dicts["tissue_thickness_sd"][edge] = tissue_thickness_sd
        measurement_dicts["total_area"][edge] = total_area_mean
        measurement_dicts["total_area_sd"][edge] = total_area_sd
        measurement_dicts["minor_axis"][edge] = minor_axis_mean
        measurement_dicts["minor_axis_sd"][edge] = minor_axis_sd
        measurement_dicts["major_axis"][edge] = major_axis_mean
        measurement_dicts["major_axis_sd"][edge] = major_axis_sd

    # Set graph attributes
    for attr, attr_dict in measurement_dicts.items():
        nx.set_edge_attributes(skeleton_graph.graph, attr_dict, name=attr)
    logger.info("save")
    # Save
    skeleton_graph.to_json_file(graph_path)

    return skeleton_graph
