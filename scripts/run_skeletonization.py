"""CLI for prediction skeleton.

To use:

python run_skeletonization.py
        --input-dir /path/to/input
        --output-dir /path/to/output
        --checkpoint checkpoint_number
        --image-key image_key
        --labels-key label_key

- input-dir: path to the directory to predict.
    All files ending with .h5 will be predicted.
- output-dir: path to the directory to save the predictions.
    The directory will be created if it does not exist.
- checkpoint: the model to use. Can be 1, 2, or 3, 4
    (these are all just trained slightly differently). 4 has updated training data.
- image-key: the key in the h5 file to use for the image.
    Default is "normalized_vector_background_image"
- labels-key: the key in the h5 file to use for the labels.
    Default is "label_image"
"""

import argparse
import glob
import os
from collections.abc import Callable

import h5py
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from morphospaces.networks.multiscale_skeletonization import (
    MultiscaleSkeletonizationNet,
)
from tqdm import tqdm


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument(
        "--image-key", type=str, default="normalized_vector_background_image"
    )
    parser.add_argument("--labels-key", type=str, default="label_image")
    args = parser.parse_args()

    return args


def load_model(
    model_class: torch.nn.Module,
    checkpoint_path: str,
    multi_gpu: bool = True,
):
    """Load a model from a checkpoint."""
    # load the network
    net = model_class.load_from_checkpoint(checkpoint_path)
    net.eval()

    if multi_gpu:
        # if we have multiple GPUs set data parallel to execute sliding window inference
        model = torch.nn.DataParallel(net)
        model = model.cuda()
    else:
        model = net.cuda()

    return model


def make_image_4d(image: np.ndarray) -> np.ndarray:
    """Make a 4D image from a 3D or 5D image."""
    if image.ndim == 3:
        return np.expand_dims(image, axis=0)
    elif image.ndim == 5:
        return np.squeeze(image, axis=0)
    else:
        raise ValueError("Image must be 3D or 5D")


def predict_image(
    image: torch.Tensor,
    predictor: Callable[[torch.Tensor], torch.Tensor],
    roi_size: tuple[int, int, int] = (120, 120, 120),
    overlap: float = 0.5,
    stitching_mode: str = "gaussian",
    progress_bar: bool = True,
    batch_size: int = 1,
) -> torch.Tensor:
    """Predict an image using sliding window inference.

    Note: image should be CZYX
    """
    # add dim -> NCZYX
    expanded_image = torch.unsqueeze(image, dim=0)

    # make the prediction
    with torch.no_grad():
        result = sliding_window_inference(
            inputs=expanded_image,
            sw_batch_size=batch_size,
            sw_device=torch.device("cuda"),
            predictor=predictor,
            roi_size=roi_size,
            overlap=overlap,
            mode=stitching_mode,
            device=torch.device("cpu"),
            progress=progress_bar,
        )

    # squeeze dims -> CZYX
    return torch.squeeze(result, dim=0)


def predict_directory(
    model: torch.nn.Module,
    input_directory_path: str,
    output_directory_path: str,
    image_key: str = "raw",
    labels_key: str = "label",
    file_pattern: str = "*.h5",
    roi_size: tuple[int, int, int] = (120, 120, 120),
    overlap: float = 0.5,
    stitching_mode: str = "gaussian",
    progress_bar: bool = True,
    batch_size: int = 1,
):
    """Predict all files in a directory using sliding window inference."""
    # make the output directory
    os.makedirs(output_directory_path, exist_ok=True)

    # get all the files
    files = glob.glob(os.path.join(input_directory_path, file_pattern))

    # predict each file
    for image_path in tqdm(files):
        # load the image
        with h5py.File(image_path, "r") as f:
            image = f[image_key][:].astype(np.single)
            labels = f[labels_key][:]

        # nans will make the network behave badly, so we will replace them with 0
        print(f"Number of nans in image: {np.sum(np.isnan(image))}")
        image[np.isnan(image)] = 0

        # make image CZYX dims
        image = torch.from_numpy(make_image_4d(image))

        result = predict_image(
            image=image,
            predictor=model,
            roi_size=roi_size,
            overlap=overlap,
            stitching_mode=stitching_mode,
            progress_bar=progress_bar,
            batch_size=batch_size,
        )

        # remove prediction outside of the labels
        cleaned_result = np.squeeze(result.numpy())
        cleaned_result[labels == 0] = 0

        # write the file
        output_path = os.path.join(output_directory_path, os.path.basename(image_path))
        with h5py.File(output_path, "w") as f_out:
            f_out.create_dataset(name=labels_key, data=labels, compression="gzip")
            f_out.create_dataset(
                name="skeleton_prediction", data=cleaned_result, compression="gzip"
            )


registered_models = {
    "1": {
        "skeletonization": "./multiscale_20231129/skel-best.ckpt",
        "autocontext": None,
    },
    "2": {
        "skeletonization": "./multiscale_skeletonization_20231116/skel-best.ckpt",
        "autocontext": None,
    },
    "3": {
        "skeletonization": "./multiscale_skeletonization_20231122/skel-best.ckpt",
        "autocontext": None,
    },
    "4": {
        "skeletonization": "./multiscale_20240214/checkpoint_20240219/skel-best.ckpt",
        "autocontext": None,
    },
}


if __name__ == "__main__":
    # get the args
    args = parse_args()
    checkpoint_path = args.checkpoint
    base_input_dir = args.input_dir
    base_output_dir = args.output_dir
    image_key = args.image_key
    labels_key = args.labels_key

    roi_size = (120, 120, 120)
    overlap = 0.5
    stitching_mode = "gaussian"
    progress_bar = True
    batch_size = 3

    if checkpoint_path in registered_models:
        model = load_model(
            model_class=MultiscaleSkeletonizationNet,
            checkpoint_path=registered_models[checkpoint_path]["skeletonization"],
            multi_gpu=True,
        )
    else:
        model = load_model(
            model_class=MultiscaleSkeletonizationNet,
            checkpoint_path=checkpoint_path,
            multi_gpu=True,
        )

    # predict directory
    predict_directory(
        model=model,
        input_directory_path=base_input_dir,
        output_directory_path=base_output_dir,
        image_key=image_key,
        labels_key=labels_key,
        file_pattern="*.h5",
        roi_size=roi_size,
        overlap=overlap,
        stitching_mode=stitching_mode,
        progress_bar=progress_bar,
        batch_size=batch_size,
    )
