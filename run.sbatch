#!/bin/bash

#SBATCH --job-name=test_sizes_final
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=128GB
#SBATCH --account=pr_95_tandon_advanced
#SBATCH --output=test_sizes_final.log
#SBATCH --error=test_sizes_final_err.log

module purge

singularity exec \
--nv --overlay /scratch/ar8692/filesystems/image-geolocation-overlay.ext3:ro \
/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
/bin/bash -c "
source /ext3/env.sh;
cd /scratch/ar8692/image-geolocation/geoclip/;
python test_sizes.py
"