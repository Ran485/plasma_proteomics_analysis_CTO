#!/bin/bash
#PBS -N MaxLFQ
#PBS -l nodes=1:ppn=4
#PBS -l walltime=120:00:00
#PBS -l mem=65gb
#PBS -j oe
#PBS -o /public/home/proteome/ranpeng/DIANN/949_cell_lines_outpath/MaxLFQ_iq_bladder.txt

# Change directory to the directory where you submitted your job
cd /public/home/proteome/ranpeng/DataStorage/xinjiang_CTO

# Activate python environment if you have one
source activate diann

# Run your python script
Rscript maxlfq_quantification.R

conda deactivate