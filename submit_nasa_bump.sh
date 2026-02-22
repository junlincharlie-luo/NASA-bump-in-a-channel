#!/bin/bash
#SBATCH --job-name=nasa_bump
#SBATCH --output=nasa_bump_%j.out
#SBATCH --error=nasa_bump_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=normal

# Load Python module (adjust version as needed)
module load python/3.9.0

# Run the NASA bump simulation
python run_nasa_bump.py --ni 200 --nj 80 --max-iter 10000 --output results

echo "Job completed at $(date)"
