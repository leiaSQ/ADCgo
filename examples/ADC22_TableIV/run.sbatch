#!/bin/bash
#SBATCH --job-name=adc22-tableIV
#SBATCH --partition=cpu-single
#SBATCH --account=bw22i008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/run-%j.log
cd /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo
export OMP_NUM_THREADS=16
go build -o /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/adcgo ./cmd/adcgo
for m in hf n2 co; do
  ~/miniconda3/envs/adcgo/bin/python scripts/fcidump/dump_fcidump.py --input /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/$m/dump.in > /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/$m/dump.log 2>&1 &
done
wait
for m in hf n2 co; do
  ( /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/adcgo -fcidump /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/$m/$m.fcidump -sip -order 22 -adc22 f -sigma off -solver dense -sym all \
      -ps-thresh 50 -coeff-thresh 0.3 -out /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/$m/adc22f.json > /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/$m/adc22f.log 2>&1 ) &
done
wait
date
