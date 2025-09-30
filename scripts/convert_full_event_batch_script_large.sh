#!/usr/bin/env bash
#SBATCH --job-name=conv_jz_large
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --output=/project/ctb-stelzer/hamza95/parnassus/logs/large_%x_%A_%a.out
#SBATCH --error=/project/ctb-stelzer/hamza95/parnassus/logs/large_%x_%A_%a.err
#SBATCH --account=def-mdanning
#SBATCH --array=1-9  # matches jz0 … jz9

# Make sure logs dir exists
mkdir -p /project/ctb-stelzer/hamza95/parnassus/logs

module load StdEnv/2023 gcc/12 python hdf5
pip install --user numpy uproot awkward tqdm

# Input and output files
INFILE="/project/ctb-stelzer/hamza95/parnassus/samples_for_parnassus/pathLists_jz${SLURM_ARRAY_TASK_ID}.txt"
OUTFILE="/project/ctb-stelzer/hamza95/parnassus/convert_full_event_large/JZ${SLURM_ARRAY_TASK_ID}.root"

echo "Processing $INFILE -> $OUTFILE"
python /project/ctb-stelzer/hamza95/parnassus/cms-flow-evt/utils/convert_full_event.py \
    -i "$INFILE" -o "$OUTFILE" -n 3000 -maxN 10000000
