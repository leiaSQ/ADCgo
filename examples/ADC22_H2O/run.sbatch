#!/bin/bash
#SBATCH --job-name=adc22-h2o
#SBATCH --partition=cpu-single
#SBATCH --account=bw22i008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/run-%j.log
cd /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo
date
for s in 0 2 3; do
  GOMAXPROCS=16 /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/adcgo -fcidump /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/h2o_apvdz.fcidump -sip -order 22 -adc22 f -solver dense -sigma off \
      -sym $s -ps-thresh 0 -coeff-thresh 1 -profile -out /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/ours_sym$s.json > /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/ours_sym$s.log 2>&1 &
done
wait
date
