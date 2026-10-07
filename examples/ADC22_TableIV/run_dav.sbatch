#!/bin/bash
#SBATCH --job-name=adc22-tableIV-dav
#SBATCH --partition=cpu-single
#SBATCH --account=bw22i008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/run_dav-%j.log
for m in n2 co; do
  ( /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/adcgo -fcidump /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/$m/$m.fcidump -sip -order 22 -adc22 f -sigma off -solver davidson -nroots 6 -convthr 1e-5 -sym all \
      -ps-thresh 50 -coeff-thresh 0.3 -out /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/$m/adc22f.json > /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/$m/adc22f_dav.log 2>&1 ) &
done
wait
date
