#!/bin/bash
#SBATCH --job-name=sigmadump
#SBATCH --partition=all
#SBATCH --mem=128G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=sigmadump-%j.out
#
# Compute ADCgo's static self-energy Σ(0,0) on the CVS core hole, on a compute node,
# and compare against theADCcode/COSTANTI's 0.936 eV. A second arg "infinite" jumps
# straight to the all-order resolvent scheme (the direct COSTANTI comparison), skipping
# the O(N^6) perturbative four/fplus that otherwise time the job out.
#
#   sbatch scripts/run_sigmadump.sbatch [fcidump] [infinite]
set -euo pipefail
here="$HOME/ADCgo"   # home is shared across nodes; $0 under slurm points at the spool copy
fc="${1:-$here/examples/CVS_NSOB/nsob_n1s.fcidump}"
scheme="${2:-infinite}"
echo "host=$(hostname)  fcidump=$fc  scheme=$scheme"
exec "$here/bin/sigmadump" "$fc" "$scheme"
