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
module load K80 impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python script.py --flag1 --flag2 > output.txt
