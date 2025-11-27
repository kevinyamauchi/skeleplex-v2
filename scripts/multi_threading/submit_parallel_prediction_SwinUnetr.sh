#! /bin/bash
#SBATCH --job-name=parallel_prediction_prep
#SBATCH --nodes=1             
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=2G

#SBATCH --time=4:00:00

python prepare_parallel_prediction.py \
./LADAF-2021-17_right_lung_scale1.zarr/image \
./LADAF-2021-17_right_lung_scale1_chunk_df.csv \
./LADAF-2021-17_right_lung_scale1_prediction.zarr/prediction


#spawn a job for each chunk
num_chunks=$(wc -l < ./LADAF-2021-17_right_lung_scale1_chunk_df.csv)
num_chunks=$((num_chunks - 1)) #subtract header line


sbatch \
    --array=0-$((num_chunks - 1)) \
    --cpus-per-task=1 \
    --mem-per-cpu=80G \
    --gpus=rtx_4090:1 \
    --time=4:00:00 \
    parallel_predict_swinUNETER.py \
    ./LADAF-2021-17_right_lung_scale1_chunk_df.csv \
    ./LADAF-2021-17_right_lung_scale1.zarr/image \
    ./LADAF-2021-17_right_lung_scale1_prediction.zarr/prediction