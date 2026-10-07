#!/bin/bash
#SBATCH --job-name=adc22-a2-fixed
#SBATCH --partition=cpu-single
#SBATCH --account=bw22i008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/a2_fixed-%j.log
/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/zz_a2_fixed /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/h2o_apvdz.fcidump 1 > /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/fixed_sym1.json
