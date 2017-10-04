#!/bin/bash
# @ job_name = jobname
# @ initialdir = /gpfs/projects/group00/user00000/scriptdir
# @ output = job_test%j.out
# @ error = job_test%j.err
# @ total_tasks = 1
# @ gpus_per_node = 1
# @ cpus_per_task = 1
# @ features = k80
# @ wall_clock_limit = 02:00:00

module purge
module load K80 cuda/8.0 mkl/2017.1 CUDNN/5.1.10-cuda_8.0 intel-opencl/2016 python/3.6.0+_ML
python script.py --flag1 --flag2 > output.txt
