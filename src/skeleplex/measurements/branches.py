import logging  # noqa
import os
import networkx as nx
import h5py
import numpy as np
import skimage as ski
import torch
from skeleplex.graph.skeleton_graph import SkeletonGraph


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from skeleplex.measurements.utils import grey2rgb
import tqdm

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
    files = tqdm.tqdm(files, desc="Processing files")
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


def add_measurements_from_h5_to_graph(
    skeleton_graph: SkeletonGraph,
    data_path: str,
):
    """
    Add measurements from h5 files to the graph.

    Parameters
    ----------
    skeleton_graph : SkeletonGraph
        The skeleton graph to add the measurements to.
    data_path : str
        Path to the directory containing the h5 files.
    """
    files = os.listdir(data_path)

    files = [f for f in files if f.endswith(".h5")]

    def radius_from_area(area):
        """Return the are of a circle based on its radius."""
        return np.sqrt(area / np.pi)

    tissue_thickness_dict = {}
    lumen_diameter_dict = {}
    total_area_dict = {}
    minor_axis_dict = {}
    major_axis_dict = {}
    for file in files:
        print(file)
        # load
        with h5py.File(data_path + file, "r") as f:
            segmentation_slices = f["segmentation"][:]

        start_node = int(file.split("_")[3])
        end_node = int(file.split("_")[5].split(".")[0])

        tissue_radius_branch = []
        lumen_radius_branch = []
        minor_axis_branch = []
        major_axis_branch = []
        total_area_branch = []
        for slice_index in range(len(segmentation_slices)):
            segmentation_slice = segmentation_slices[slice_index]
            if np.sum(segmentation_slice == 2) > 0:
                # open
                lumen_label = (segmentation_slice == 2) * 1
                tissue_label = (segmentation_slice == 1) * 1

                lumen_props = ski.measure.regionprops(ski.measure.label(lumen_label))
                tissue_props = ski.measure.regionprops(ski.measure.label(tissue_label))
                if len(lumen_props) > 0:
                    # get the one with biggest area
                    lumen_props = sorted(
                        lumen_props, key=lambda x: x.area, reverse=True
                    )
                    lumen_props = [lumen_props[0]]
                if len(tissue_props) > 0:
                    # get the one with biggest area
                    tissue_props = sorted(
                        tissue_props, key=lambda x: x.area, reverse=True
                    )
                    tissue_props = [tissue_props[0]]

                if len(tissue_props) == 0:
                    logger.info(f"no tissue props in slice {slice_index}, file {file}")
                    continue

                lumen_area = lumen_props[0].area
                tissue_area = tissue_props[0].area
                total_area = lumen_area + tissue_area
                total_radius = radius_from_area(total_area)
                lumen_radius = radius_from_area(lumen_area)
                tissue_radius = total_radius - lumen_radius
                tissue_radius_branch.append(tissue_radius)
                lumen_radius_branch.append(lumen_radius)
                minor_axis = tissue_props[0].minor_axis_length
                major_axis = tissue_props[0].major_axis_length
                total_area_branch.append(total_area)

            else:
                # closed
                label_slice = ski.measure.label((segmentation_slice != 0) * 1)
                props = ski.measure.regionprops(label_slice)
                # get the one with biggest area
                if len(props) > 0:
                    props = sorted(props, key=lambda x: x.area, reverse=True)
                    props = [props[0]]
                minor_axis = props[0].minor_axis_length
                major_axis = props[0].major_axis_length
                total_area = props[0].area
                tissue_radius_branch.append(minor_axis / 2)
                lumen_radius_branch.append(0)
                total_area_branch.append(total_area)
                minor_axis_branch.append(minor_axis)
                major_axis_branch.append(major_axis)

        tissue_thickness_dict[(start_node, end_node)] = np.mean(tissue_radius_branch)
        lumen_diameter_dict[(start_node, end_node)] = 2 * np.mean(lumen_radius_branch)
        total_area_dict[(start_node, end_node)] = np.mean(total_area_branch)
        minor_axis_dict[(start_node, end_node)] = np.mean(minor_axis_branch)
        major_axis_dict[(start_node, end_node)] = np.mean(major_axis_branch)

        nx.set_edge_attributes(
            skeleton_graph.graph, tissue_thickness_dict, name="tissue_thickness"
        )
        nx.set_edge_attributes(
            skeleton_graph.graph, lumen_diameter_dict, name="lumen_diameter"
        )
        nx.set_edge_attributes(skeleton_graph.graph, total_area_dict, name="total_area")
        nx.set_edge_attributes(skeleton_graph.graph, minor_axis_dict, name="minor_axis")
        nx.set_edge_attributes(skeleton_graph.graph, major_axis_dict, name="major_axis")
