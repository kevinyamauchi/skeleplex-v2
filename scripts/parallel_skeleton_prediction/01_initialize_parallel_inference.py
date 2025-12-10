"""Demo script for initializing parallel skeleton inference using SLURM.

This script sets up the chunk table and generates the SLURM sbatch command.
"""

from pathlib import Path

from skeleplex.utils import build_sbatch_command, initialize_parallel_inference

if __name__ == "__main__":
    # path to the file containing the distance field image
    distance_field_path = Path("demo_tubes.zarr/distance_field")

    # path to save the skeleton output
    skeleton_output_path = Path("demo_tubes.zarr/skeleton")

    # path to the chunk table
    chunk_table_path = Path("chunk_table.csv")

    # slurm parameters
    job_name = "skeleton_inference"
    time_limit = "00:05:00"  # hh:mm:ss
    memory = "16G"  # memory per CPU
    cpus_per_task = 1
    n_gpus = 1
    gpu_name = "rtx_4090"
    run_command = "python 02_infer_skeleton_chunk.py --chunk-index $SLURM_ARRAY_TASK_ID"

    # size of the chunks to process in parallel
    # these must be an integer multiple of the
    # chunks in the distance field zarr
    processing_chunk_size = (40, 40, 40)

    # size of the border to include around each chunk
    border_size = (20, 20, 20)

    n_chunks = initialize_parallel_inference(
        input_zarr_path=distance_field_path,
        output_zarr_path=skeleton_output_path,
        chunk_shape=processing_chunk_size,
        border_size=border_size,
        chunk_table_path=chunk_table_path,
    )

    slurm_command = build_sbatch_command(
        n_array_jobs=n_chunks,
        job_name=job_name,
        time_limit=time_limit,
        memory=memory,
        cpus_per_task=cpus_per_task,
        n_gpus=n_gpus,
        gpu_name=gpu_name,
        run_command=run_command,
    )

    print("Submit the following command to SLURM to start the inference:")
    print(slurm_command)
