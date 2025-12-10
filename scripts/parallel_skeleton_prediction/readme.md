# parallel inference example

This example demonstrates how to perform parallel inference with the skeletonization model using a slurm job array. The strategy is to divide the input volume into chunks, and assign each chunk to a separate job in the job array. Each job will load the model, run inference on its assigned chunk, and save the results. This is achieved used the following scripts:

- 00_make_demo_image.py: a script to create the demo zarr file with synthetic data (tubes)
- 01_initialize_parallel_inference.py: a script to create the chunk table used to determine which chunk to run per array job. it also prepares the slurm command to run the job array.
- 02_infer_skeleton_chunk.py: this script is called by each slurm array job to infer the skeleton for its assigned chunk. You do not need to call this script directly, it is called by slurm.

## Instructions
0. Activate your skeleplex environment.
1. Run the `00_make_demo_image.py` script to create the demo zarr file with synthetic data.
2. Run the `01_initialize_parallel_inference.py` script to create the chunk table and prepare the slurm command. This will print out the slurm command to run the job array. Example output:

```bash
Submit the following command to SLURM to start the inference:
sbatch --job-name=skeleton_inference --array=0-26 --time=00:05:00 --cpus-per-task=1 --mem-per-cpu=16G --gpus=rtx_4090:1 --wrap='python 02_infer_skeleton_chunk.py --chunk-index $SLURM_ARRAY_TASK_ID'
```

3. Copy and paste the printed slurm command into your terminal to submit the job array to the slurm scheduler.

## Modifying for your use case
To adapt this example for your own data and model, you will need to modify the following parameters in the scripts.

**01_initialize_parallel_inference.py**
- `distance_field_path`: path to your input distance field zarr array.
- `skeleton_output_path`: path where you want to save the output skeleton zarr array.
- Update the slurm parameters to meet the needs of your array. Do not modify the `run_command`.
- `processing_chunk_size`: size of the chunks to divide the input volume into. Adjust based on your data size and available resources. This should be an integer multiple of the input array's chunk size. For example, if the input array has chunk size (25, 35, 45), you could use (50, 70, 90) or (75, 105, 135), etc. This chunk does not need to fit into GPU memory, as each chunk is inferred in smaller sub-chunks using sliding window inference. The sub-chunk size is set in the `02_infer_skeleton_chunk.py` script.
- `border_size`: size of the border to add around each chunk to avoid edge artifacts during inference. Adjust based on your model's receptive field.

**02_infer_skeleton_chunk.py**
- `roi_size`: size of the sub-chunks to use during inference. This should fit into your GPU memory along with the model. Adjust based on your model size and available GPU memory.
- `batch_size`: number of sub-chunks to process in a single batch during inference. Adjust based on your GPU memory and model size.
- `overlap`: the fractional overlaps of the sub-chunks for the sliding window inference. Adjust based on your model's receptive field.



