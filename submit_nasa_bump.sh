#!/bin/bash
#SBATCH --job-name=nasa_bump
#SBATCH --output=nasa_bump_%j.out
#SBATCH --error=nasa_bump_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=normal

# Load modules
module purge
module load python/3.9.0
module load py-numpy/1.24.2_py39
module load py-matplotlib/3.7.1_py39

# Print environment for debugging
echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo "Files:"
ls -la

# Run the NASA bump simulation
python run_nasa_bump.py --ni 200 --nj 80 --max-iter 10000 --output results

echo "Job completed at $(date)"
echo "Results:"
ls -la results/ 2>/dev/null || echo "No results directory created"
