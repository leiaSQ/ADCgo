#!/bin/bash
#SBATCH --job-name=adc22-h2o-fixed
#SBATCH --partition=cpu-single
#SBATCH --account=bw22i008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/fixed_dense-%j.log
B=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/adcgo_fix
for s in 0 1 2 3; do
  GOMAXPROCS=16 $B -fcidump /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/h2o_apvdz.fcidump -sip -order 22 -adc22 f -solver dense -sigma off \
      -sym $s -ps-thresh 0 -coeff-thresh 1 -out /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/fixed_sym$s.json > /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/fixed_sym$s.log 2>&1 &
done
wait
date
