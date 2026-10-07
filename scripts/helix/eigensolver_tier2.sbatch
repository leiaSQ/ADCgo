#!/bin/bash
# Production-BAND measurement of the Mode B banded eigensolve: band = 3079, the melanin DIP
# half-bandwidth, at a dim small enough to finish in a job.
#
# This is the only measurement that reproduces the regime that matters. Tier 1 ({3712,37}) has
# production's chase length but 74 accumulated rows against production's 3080, and its arrays fit
# in L3; band = 3079 puts the per-column footprint at ~39 MB and the accumulated rows at their real
# width, which is what decides both the replay's throughput and how many workers it can use.
#
# It is still NOT production's dim (308000). Cost is linear in dim at fixed band, so run two dims a
# factor 2 apart and check the ratio: worse than 2x means DRAM effects have begun and the
# extrapolation to production is a lower bound on the time.
#SBATCH --account=bw22i008
#SBATCH --job-name=eig-tier2
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=180gb
#SBATCH --time=05:00:00
#SBATCH --output=eig-tier2-%j.out
set -u
root=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo
cd "$root"
# Save the toolchain's bin dir before any module load: compiler/gnu puts gccgo 1.16 ahead of Go.
export PATH="$(dirname "$(command -v go)"):$PATH"
export GOMAXPROCS=${SLURM_CPUS_PER_TASK:-64}
export TIER2_DIMS=${TIER2_DIMS:-9240}
echo "host=$(hostname) date=$(date -Is) cpus=$SLURM_CPUS_PER_TASK GOMAXPROCS=$GOMAXPROCS TIER2_DIMS=$TIER2_DIMS"
go version
go test ./internal/adc/lanczos/ -run XXX -bench BenchmarkBandSymDiagFastTier2 -benchtime 1x -timeout 290m
