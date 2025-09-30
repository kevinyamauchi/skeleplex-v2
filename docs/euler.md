# Running SkelePlex on ETH Euler Cluster

## Making the Environment on Euler

1. Connect to the ETH network, either by using the on-side wifi/Ethernet or by using a VPN
2. Start a terminal
3. Use ssh command to connect to the login node of Euler: `ssh username@euler.ethz.ch`, use your regular ETH password to connect
4. First use only: clone SkelePlex-V2 to your working directory (e.g., /cluster/home/username/), you might have to add an SHH key for this step, follow tutorial on:[add SHH key GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
5. cd into the folder where SkelePlex-V2 is installed 
6. Create a new virtual environment using these commands:
    1. Load module stack with pre-installed packages: `module load stack/2024-06 python_cuda/3.11.6`
    2. Create new virtual environment called skeleplexenv: `python -m venv --system-site-packages skeleplexenv`
    3. Activate your new virtual environment called skeleplexenv: `source skeleplexenv/bin/activate`
    4. Install skeleplex into that virtual environment:`pip install -e ".[dev-all]"`
    5. Install cuda into that virtual environment:`pip install cupy-cuda12x`



## Send your Data to Euler

This is how to upload a file to a specific folder on Euler, for large .zarr files it is recommended to send them in a compressed format and then unzip on Euler.
1. Sent to Euler: `rsync -av  /path/to/file/yourimage.zarr.zip  username@euler.ethz.ch:/cluster/where/you/save/data`
2. Unzip file on Euler: `unzip yourimage.zarr.zip`


## Write and Run the Job Script on Euler
- Create a new file containing your skeleplex job: `nano skeleplex_job.sh`
- Write the Job Script, set the requirements according to your needs, use the [HPC - Slurm Submission Line Advisor](https://docs.hpc.ethz.ch/services/slurm-submission-line-advisor/) for the set-up. See an example below:

    ```#!/bin/bash

    #SBATCH --job-name=skeleplex_job
    #SBATCH --time=04:00:00
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem-per-cpu=64G
    #SBATCH --gpus=rtx_4090:1     
    #SBATCH --output=skeleplex_job_%j.out

    python skeleplex_job.py --workers 1

    echo "Job completed: $(date)"```

- Save the file (`Ctrl+X`, then `Y`, then `Enter` in nano).
- To ensure that SkelePlex can be accessed run the following command before submitting your job: `module load eth_proxy`
- Then, submit the job: `sbatch skeleplex_job.sh`. You will obtain a job ID that you can use to look into the status of your submission: `squeue -j 12345678` (replace with your actual job ID) or `myjobs -j 12345678` for more detailed information (for both commands, replace with actual job ID).
- You can view the output file with this line: `cat skeleplex_job_12345678.out` (replace with actual job ID).

## Retrieve Data from Euler
This is how you can pull back the output files to your local computer: `rsync -av username@euler.ethz.ch:cluster/where/you/save/data/*.zarr /path/to/save/file/to/here/`


## Resources used in above Text: 

- [HPC - first job on Euler](https://docs.hpc.ethz.ch/tutorials/first-job/)
- [HPC - Slurm Submission Line Advisor](https://docs.hpc.ethz.ch/services/slurm-submission-line-advisor/) 
- [Add SHH key GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
