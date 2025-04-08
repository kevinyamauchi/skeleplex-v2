import warnings
from typing import Literal

import numpy as np
import torch

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from monai.inferers import sliding_window_inference
from morphospaces.networks import MultiscaleSemanticSegmentationNet

from skeleplex.skeleton._utils import make_image_5d


def segment(
    image: np.ndarray,
    model: Literal["pretrained"] | MultiscaleSemanticSegmentationNet = "pretrained",
    roi_size: tuple[int, int, int] = (120, 120, 120),
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

    # get the skeletonziation model if requested
    if model == "pretrained":
        raise NotImplementedError("pretrained segmentation models not implemented.")

    # put the model in eval mode
    model.eval()

    # make the prediction
    with torch.no_grad():
        result = sliding_window_inference(
            inputs=expanded_image,
            sw_batch_size=batch_size,
            sw_device=torch.device("cuda"),
            predictor=model,
            roi_size=roi_size,
            overlap=overlap,
            mode=stitching_mode,
            device=torch.device("cpu"),
            progress=progress_bar,
        )

    # squeeze dims -> ZYX
    return torch.squeeze(torch.squeeze(result, dim=0), dim=0).numpy()
