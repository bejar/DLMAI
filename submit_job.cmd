#!/bin/bash
#SBATCH --job-name=NAME_OF_JOB
#SBATCH -D/path/to/the/code
#SBATCH --output=output_%j.out
#SBATCH --error=errors_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=HH:MM:SS
#SBATCH --gres=gpu:1
#SBATCH --mem=MEMORY

module purge
module load  gcc/6.4.0  cuda/9.1 cudnn/7.1.3 openmpi/3.0.0 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.7 szip/2.1.1 ffmpeg/4.0.2 opencv/3.4.1 python/3.6.5_ML
PYTHONPATH={jobs_code_path}
export PYTHONPATH

python script.py --flag1 --flag2 > std_output.txt
