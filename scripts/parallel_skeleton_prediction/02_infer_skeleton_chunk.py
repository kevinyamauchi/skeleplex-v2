"""Demo script for performing skeletonization inference on a data chunk.

This uses the data chunk information from a CSV file to perform inference.
"""

import argparse
from functools import partial

import pandas as pd

from skeleplex.skeleton import skeletonize
from skeleplex.utils import infer_on_chunk

# make a CLI argument parser for --job-index
parser = argparse.ArgumentParser()
parser.add_argument(
    "--chunk-index",
    type=int,
    required=True,
    help="Index of the chunk to process.",
)


if __name__ == "__main__":
    # path to the chunk table
    chunk_table_path = "chunk_table.csv"

    # roi size for the sliding window inference
    roi_size = (40, 40, 40)
    batch_size = 1
    overlap = 0.5

    # parse the arguments
    arguments = parser.parse_args()
    chunk_index = arguments.chunk_index

    # get the chunk data from the table
    chunk_table = pd.read_csv(chunk_table_path)
    chunk_info = chunk_table.iloc[chunk_index]
    input_zarr_path = chunk_info["input_zarr_path"]
    output_zarr_path = chunk_info["output_zarr_path"]

    # Extract core chunk boundaries
    z_start = int(chunk_info["z_start"])
    z_end = int(chunk_info["z_end"])
    y_start = int(chunk_info["y_start"])
    y_end = int(chunk_info["y_end"])
    x_start = int(chunk_info["x_start"])
    x_end = int(chunk_info["x_end"])
    border_z = int(chunk_info["border_z"])
    border_y = int(chunk_info["border_y"])
    border_x = int(chunk_info["border_x"])

    # make the inference function
    inference_function = partial(
        skeletonize,
        roi_size=roi_size,
        overlap=overlap,
        batch_size=batch_size,
    )

    # infer on the chunk
    infer_on_chunk(
        input_zarr_path=input_zarr_path,
        output_zarr_path=output_zarr_path,
        inference_function=inference_function,
        z_start=z_start,
        z_end=z_end,
        y_start=y_start,
        y_end=y_end,
        x_start=x_start,
        x_end=x_end,
        border_z=border_z,
        border_y=border_y,
        border_x=border_x,
    )
