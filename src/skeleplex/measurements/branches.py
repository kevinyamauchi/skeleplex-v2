import logging  # noqa
import os

import h5py
import numpy as np
import skimage.measure as ski
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from skeleplex.measurements.utils import grey2rgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def filter_and_segment_lumen(
    data_path,
    save_path,
    sam_checkpoint_path,
    eccentricity_thresh=0.7,
    circularity_thresh=0.5,
):
    """
    Filter and segment the lumen in the image slices.

    Parameters
    ----------
    data_path : str
        Path to the input data directory containing .h5 files.
    save_path : str
        Path to the output directory where filtered .h5 files will be saved.
    sam_checkpoint_path : str
        Path to the SAM2 checkpoint file.
    eccentricity_thresh : float
        Eccentricity threshold for filtering slices.
    circularity_thresh : float
        Circularity threshold for filtering slices.


    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logger.info(f"Created directory: {save_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam2_checkpoint = sam_checkpoint_path
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files = [f for f in os.listdir(data_path) if f.endswith(".h5")]

    for file in files:
        logger.info(f"Processing {file}")

        with h5py.File(os.path.join(data_path, file), "r") as f:
            image_slices = f["image"][:]
            segmentation_slices = f["segmentation"][:] != 0

        # start_node = int(file.split('_')[3])
        # end_node = int(file.split('_')[5].split('.')[0])

        label_slices_filt = np.zeros_like(segmentation_slices, dtype=np.uint8)
        index_to_remove = []

        for i in range(len(image_slices)):
            image_slice = image_slices[i]
            segmentation_slice = segmentation_slices[i]
            label_slice = ski.label(segmentation_slice)
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
            props = ski.regionprops(label_slice)
            if props[0].eccentricity > eccentricity_thresh:
                index_to_remove.append(i)
                continue

            # Check circularity
            circularity = 4 * np.pi * props[0].area / (props[0].perimeter ** 2)
            if circularity < circularity_thresh:
                index_to_remove.append(i)
                continue

            # Segment using SAM2
            image_slice_rgb = grey2rgb(image_slice)
            predictor.set_image(image_slice_rgb)
            sam_point = np.array([[h // 2, w // 2]])
            sam_label = np.array([1])
            sam_mask, _, _ = predictor.predict(
                point_coords=sam_point, point_labels=sam_label, multimask_output=False
            )

            label_with_lumen = label_slice.copy()
            label_with_lumen[sam_mask[0] == 1] = 2

            # Ensure lumen segmentation is valid
            lumen_mean = image_slice[label_with_lumen == 2].mean()
            label_mean = image_slice[label_with_lumen == 1].mean()
            if lumen_mean < label_mean:
                label_slice = label_with_lumen
            else:
                label_slice = label_slice

            label_slices_filt[i] = label_slice

        # Remove invalid slices
        image_slice_filt = np.delete(image_slices, index_to_remove, axis=0)
        label_slices_filt = np.delete(label_slices_filt, index_to_remove, axis=0)

        # Save filtered data
        with h5py.File(os.path.join(save_path, file), "w") as f:
            f.create_dataset("image", data=image_slice_filt)
            f.create_dataset("segmentation", data=label_slices_filt)
