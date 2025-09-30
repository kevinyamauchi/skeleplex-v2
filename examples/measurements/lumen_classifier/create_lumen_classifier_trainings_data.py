import os  # noqa: D100

import h5py
import napari
import numpy as np
import skimage as ski
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage import measure
from tqdm import tqdm

from skeleplex.measurements.lumen_classifier import SaveClassifiedSlices
from skeleplex.measurements.utils import grey2rgb

# should be a subset of slices to train the classifier on
# Ideally balanced between classes
# Here we just use all slices from the example data
data_path = "../../example_data/branch_slices.h5"

files = [f for f in os.listdir(data_path) if f.endswith(".h5")]
# only do those that are npot in the save_path
files = tqdm(files, desc="Processing files")

# Load SAM2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(model)

viewer = napari.Viewer()
viewer.window.add_dock_widget(SaveClassifiedSlices(viewer), area="right")


for file in files:
    with h5py.File(os.path.join(data_path, file), "r") as f:
        image_slices = f["image"][:]
        segmentation_slices = f["segmentation"][:] != 0

    label_slices_filt = np.zeros_like(segmentation_slices, dtype=np.uint8)
    sam_pred_slices = []
    cropped_image_slices = []
    # Go through each slice
    for i in range(len(image_slices)):
        image_slice = image_slices[i]
        segmentation_slice = segmentation_slices[i]
        label_slice = ski.measure.label(segmentation_slice)
        h, w = image_slice.shape
        central_label = label_slice[h // 2, w // 2]

        # Skip if no central label
        if central_label == 0:
            continue

        label_slice[label_slice != central_label] = 0
        label_slice[label_slice == central_label] = 1

        # Segment using SAM2
        image_slice_rgb = grey2rgb(image_slice)
        predictor.set_image(image_slice_rgb)
        sam_point = np.array([[h // 2, w // 2]])
        sam_label = np.array([1])
        sam_mask, _, _ = predictor.predict(
            point_coords=sam_point,
            point_labels=sam_label,
            multimask_output=False,
        )

        mask_img = image_slice * sam_mask[0]

        sam_pred_slices.append(sam_mask[0].astype(int))
        cropped_image_slices.append(mask_img)

    cropped_stack = np.stack(cropped_image_slices)
    sam_stack = np.stack(sam_pred_slices)

    # crop to bounding box
    min_z, min_y, min_x, max_z, max_y, max_x = measure.regionprops(
        sam_stack.astype(int)
    )[0].bbox

    # Crop in 3D
    sam_mask = sam_stack[min_z:max_z, min_y:max_y, min_x:max_x].astype(int)
    mask_img = cropped_stack[min_z:max_z, min_y:max_y, min_x:max_x]

    viewer.add_image(mask_img, name=f"{file}.h5_img", visible=False)
    viewer.add_labels(sam_mask, name=f"{file}.h5_sam", visible=False)

    # Now set your lumen, branch and bad paths and classify away
    # until you have enough training data
    # Then you can train your ResNet classifier on the saved data

napari.run()
