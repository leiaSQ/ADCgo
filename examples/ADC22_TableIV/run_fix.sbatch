#!/bin/bash
#SBATCH --job-name=adc22-symfix
#SBATCH --partition=cpu-single
#SBATCH --account=bw22i008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/run_fix-%j.log
cd /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo
go build -o /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/adcgo_fix ./cmd/adcgo || exit 1
B=/gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/adcgo_fix
run() { # fcidump out
  $B -fcidump $1 -sip -order 22 -adc22 f -sigma off -solver davidson -nroots 6 -convthr 1e-5 -sym all \
    -ps-thresh 50 -coeff-thresh 0.3 -out $2 > ${2%.json}.log 2>&1 &
}
run /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/h2o_apvdz.fcidump /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_H2O/fixed_c2v.json
run /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/co/co.fcidump /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/co/adc22f_fixed.json
run /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/n2/n2.fcidump /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/n2/adc22f_fixed.json
run /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/hf/hf.fcidump /gpfs/bwfor/home/hd/hd_hd/hd_hh323/ADCgo/examples/ADC22_TableIV/hf/adc22f_fixed.json
wait
date
