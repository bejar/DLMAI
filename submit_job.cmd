#SBATCH --job-name="jobname"
#SBATCH -D/gpfs/projects/group00/user00000/scriptdir
#SBATCH --output=/gpfs/projects/group00/user00000/scriptdir/output.out
#SBATCH --error=/gpfs/projects/group00/user00000/scriptdir/errors.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:50:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
module purge
module load K80 impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python script.py --flag1 --flag2 > std_output.txt
