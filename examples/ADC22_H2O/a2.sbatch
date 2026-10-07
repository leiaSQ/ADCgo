#!/bin/bash
#SBATCH --job-name=adc22-h2o-all
#SBATCH --partition=cpu-single
#SBATCH --account=bw22i008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/all-%j.log
cd /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O
for s in 0 1 2 3; do GOMAXPROCS=16 ./zz_a2 h2o_apvdz.fcidump $s > full_sym$s.json & done
wait
date
