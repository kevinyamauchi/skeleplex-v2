import logging  # noqa
import os
import h5py

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def write_slices_to_h5(
    file_path: str,
    file_base: str,
    image_slices: dict,
    segmentation_slices: dict | None = None,
    image_key: str = "image",
    segmentation_key: str = "segmentation",
):
    """
    Write image and segmentation slices to an h5 file.

    Parameters
    ----------
    file_path : str
        The path to save to.
    file_base : str
        The base name of the file.
    image_slices : dict
        A dictionary of image slices. Keys are the edge IDs
        of the edge the image slice belongs to.
    segmentation_slices : dict, optional
        A dictionary of segmentation slices. Keys are the edge IDs
        of the edge the segmentation slice belongs to.
    image_key : str
        The key to use for the image slices.
    segmentation_key : str
        The key to use for the segmentation slices.
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    for edge in image_slices.keys():
        file_name = os.path.join(file_path, f"{file_base}_sn_{edge[0]}_en_{edge[1]}.h5")
        logger.info(f"Writing edge {edge} to {file_name}")

        with h5py.File(file_name, "w") as f:
            f.create_dataset(image_key, data=image_slices[edge])
            if segmentation_slices is not None:
                f.create_dataset(segmentation_key, data=segmentation_slices[edge])
