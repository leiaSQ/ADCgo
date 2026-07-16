#!/bin/bash
#SBATCH --job-name=sigmainf
#SBATCH --partition=all
#SBATCH --mem=120G
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=sigmainf-%j.out
set -euo pipefail
here="$HOME/ADCgo"
fc="${1:-$here/examples/CVS_NSOB/nsob_n1s.fcidump}"
echo "host=$(hostname)  fcidump=$fc  (infinite-only)"
exec "$here/bin/sigmadump" "$fc" infinite
