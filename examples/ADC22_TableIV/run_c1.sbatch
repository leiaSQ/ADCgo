#!/bin/bash
#SBATCH --job-name=adc22-c1
#SBATCH --partition=cpu-single
#SBATCH --account=bw22i008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=230G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/run_c1-%j.log
cd /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo
export OMP_NUM_THREADS=16
B=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/adcgo
run() { # dir fcidump nroots
  ( cd $1 && ~/miniconda3/envs/adcgo/bin/python /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/scripts/fcidump/dump_fcidump.py --input $1/dump.in > $1/dump.log 2>&1 && \
    $B -fcidump $1/$2 -sip -order 22 -adc22 f -sigma off -solver davidson -nroots $3 -convthr 1e-5 -sym all \
      -ps-thresh 50 -coeff-thresh 0.3 -out $1/adc22f.json > $1/adc22f.log 2>&1 ) &
}
run /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/c1 h2o_c1.fcidump 8
run /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/co_c1 co.fcidump 10
wait
date
