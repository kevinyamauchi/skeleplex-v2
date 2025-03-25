import numpy as np
import torch
from monai.inferers import sliding_window_inference

from skeleplex.skeleton._utils import get_skeletonization_model, make_image_5d


def skeletonize(
    image: np.ndarray,
    roi_size: tuple[int, int, int] = (120, 120, 120),
    overlap: float = 0.5,
    stitching_mode: str = "gaussian",
    progress_bar: bool = True,
    batch_size: int = 1,
) -> np.ndarray:
    """Skeletonize a normalized distance field image.

    Parameters
    ----------
    image : np.ndarray
        The input image to skeletonize.
        This should be a normalized distance field image.
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

    # get the skeletonziation model
    model = get_skeletonization_model()
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
