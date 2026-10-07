#!/bin/bash
#SBATCH --job-name=adc22-h2o-var
#SBATCH --partition=cpu-single
#SBATCH --account=bw22i008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/variants-%j.log
cd /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O
for v in m x; do for s in 0 1 2 3; do GOMAXPROCS=8 ./zz_adc22 h2o_apvdz.fcidump $s $v > full_${v}_sym$s.json & done; done
for s in 0 2 3; do GOMAXPROCS=8 ./adcgo -fcidump h2o_apvdz.fcidump -sip -order 22 -adc22 f -solver dense -sigma auto -sym $s -ps-thresh 1 -coeff-thresh 1 -out sigma_sym$s.json > sigma_sym$s.log 2>&1 & done
wait
date
