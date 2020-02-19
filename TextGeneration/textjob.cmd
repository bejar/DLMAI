#!/bin/bash
#SBATCH --job-name=TextGeneration
#SBATCH -D/gpfs/projects/bsc28/bsc28642/DL/DLMAI/TextGeneration
#SBATCH --output=job_text_%j.out
#SBATCH --error=job_text_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8000

module purge
module load  gcc/6.4.0  cuda/9.1 cudnn/7.1.3 openmpi/3.0.0 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.7 szip/2.1.1 ffmpeg/4.0.2 opencv/3.4.1 python/3.6.5_ML
PYTHONPATH={/gpfs/projects/bsc28/bsc28642/DL/DLMAI/TextGeneration}
export PYTHONPATH

python TextGenerator.py --verbose --progress --save --iterations 1 --neurons 16 --layers 1 --exlen 10 --step 1 --datafile 1