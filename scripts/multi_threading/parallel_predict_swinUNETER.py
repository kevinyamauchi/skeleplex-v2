#!/usr/bin/env python  # noqa: D100
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import zarr
from morphospaces.networks.swin_unetr import StandardSwinUNETR

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from monai.inferers import SlidingWindowInfererAdapt

from skeleplex.skeleton._utils import make_image_5d


def parse_slice_string(slice_str: str) -> tuple:
    """Convert string representation of slices back to tuple of slice objects.

    Parameters
    ----------
    slice_str : str
        String like "(slice(x, y, None), slice(x, y, None), slice(x, y, None))"

    Returns
    -------
    tuple
        Tuple of slice objects
    """
    # Find all slice(...) patterns
    slice_pattern = r'slice\(([^)]+)\)'
    matches = re.findall(slice_pattern, slice_str)

    slices = []
    for match in matches:
        # Parse the arguments: start, stop, step
        args = match.split(',')
        args = [arg.strip() for arg in args]

        # Convert 'None' string to None, otherwise to int
        parsed_args = []
        for arg in args:
            if arg == 'None':
                parsed_args.append(None)
            else:
                parsed_args.append(int(arg))

        # Create slice object
        if len(parsed_args) == 2:
            slices.append(slice(parsed_args[0], parsed_args[1]))
        elif len(parsed_args) == 3:
            slices.append(slice(parsed_args[0], parsed_args[1], parsed_args[2]))
        else:
            slices.append(slice(parsed_args[0]))

    return tuple(slices)

def get_dicts_from_dataframe(job_dataframe: pd.DataFrame):
    """Extracts chunking dictionaries from a job dataframe."""
    # Convert string representations back to slice tuples
    input_slices_dict = {
        k: parse_slice_string(v)
        for k, v in job_dataframe['input_slices'].to_dict().items()
    }
    expanded_slices_dict = {
        k: parse_slice_string(v)
        for k, v in job_dataframe['expanded_slices'].to_dict().items()
    }
    core_in_result_slices_dict = {
        k: parse_slice_string(v)
        for k, v in job_dataframe['core_in_result_slices'].to_dict().items()
    }
    core_in_result_slices_extended_dict = {
        k: parse_slice_string(v)
        for k, v in job_dataframe['core_in_result_slices_extended'].to_dict().items()
    }

    return (
        input_slices_dict,
        expanded_slices_dict,
        core_in_result_slices_dict,
        core_in_result_slices_extended_dict
    )
def segment(
    image: np.ndarray,
    model: StandardSwinUNETR,
    roi_size: tuple[int, int, int] = (192, 192, 192),
    overlap: float = 0.5,
    stitching_mode: str = "gaussian",
    progress_bar: bool = True,
    batch_size: int = 1,
) -> np.ndarray:
    """Segment the structures to be skeletonized.

    In the case of lungs, this would be used to segment the airways.

    Parameters
    ----------
    image : np.ndarray
        The input image to skeletonize.
        This should be a normalized distance field image.
    model : Literal["pretrained"] | MultiscaleSemanticSegmentationNet = "pretrained",
        The model to use for prediction. This can either be an instance of
        MultiscaleSemanticSegmentationNet or the string "pretrained". If "pretrained",
        a pretrained model will be downloaded from the SkelePlex repository and used.
        Default value is "pretrained".
    roi_size : tuple[int, int, int]
        The size of each tile to predict on.
        The default value is (120, 120, 120).
    overlap : float
        The amount of overlap between tiles.
        Should be between 0 and 1.
        Default value is 0.5.
    stitching_mode : str
        The method to use to stitch overlapping tiles.
        Should be "gaussian" or "constant".
        "gaussian" uses a Gaussian kernel to weight the overlapping regions.
        "constant" uses equal weight across overlapping regions.
        "gaussian" is the default.
    progress_bar : bool
        Displays a progress bar during the prediction when set to True.
        Default is True.
    batch_size : int
        The number of tiles to predict at once.
        Default value is 1.
    """
    # add dim -> NCZYX
    expanded_image = torch.from_numpy(make_image_5d(image))

    # put the model in eval mode
    model.eval()
    # inferer = SimpleInferer()
    inferer = SlidingWindowInfererAdapt(
        roi_size=roi_size,
        sw_device=torch.device("cuda"),
        sw_batch_size=batch_size,
        overlap=overlap,
        mode=stitching_mode,
        progress=progress_bar,
    )
    # make the prediction
    with torch.no_grad():
        result = inferer(inputs=expanded_image, network=model.forward_infer)


    # squeeze dims -> ZYX
    result_cpu = result.cpu()

    del result
    #only keep the second channel
    return torch.squeeze(torch.squeeze(result_cpu, dim=0), dim=0).numpy()[1]




if __name__ == "__main__":

    ckpt_path = "checkpoints_semantic_UNETR_v3/seg-best-v1.ckpt"

    chunk_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    job_df  = pd.read_csv(sys.argv[1])
    input_path = sys.argv[2]
    output_path = sys.argv[3]


    roi_size = (192, 192, 192)
    sw_overlap = 0.5
    batch_size = 1

    #get chunking dicts
    (
        input_slices_dict,
        expanded_slices_dict,
        core_in_result_slices_dict,
        core_in_result_slices_extended_dict
    ) = get_dicts_from_dataframe(job_df)

    expanded_slices = expanded_slices_dict[chunk_id]
    core_in_result_slices = core_in_result_slices_dict[chunk_id]
    core_in_result_slices_extended = core_in_result_slices_extended_dict[chunk_id]

    image = zarr.open(input_path, mode='r')[expanded_slices]

    model = StandardSwinUNETR.load_from_checkpoint(ckpt_path)
    # Zarr store
    output = zarr.open(output_path,
                            mode="a",)

    prediction = segment(
        image=image,
        model=model,
        roi_size=roi_size,
        overlap=sw_overlap,
        stitching_mode="gaussian",
        progress_bar=True,
        batch_size=batch_size,
    )
    output[core_in_result_slices_extended] = prediction[core_in_result_slices]

    print(f"Finished chunk {chunk_id}.")



