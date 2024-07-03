#!/bin/bash
# use conda
# use docker
# file: MSConvert_rawdata.sh
#PBS -N Convert rawdata
#PBS -l nodes=1:ppn=1
##PBS -l mem=10gb
#PBS -l walltime=120:00:00
#PBS -q default
#PBS -j oe
# import all environments virables
#PBS -V

inputDIR="/public/home/proteome/ranpeng/DataStorage/xinjiang_CTO/CTO_raw"

singularity exec \
        --bind /public/group_share_data/proteome/ranpeng/:/public/group_share_data/proteome/ranpeng/ \
        -B $inputDIR:/data \
        -S /mywineprefix/ \
        /public/home/proteome/ranpeng/RT_Normalize/RanPeng/pwiz-skyline-i-agree-to-the-vendor-licenses_latest.sif \
        mywine msconvert --zlib --32 \
                        /data/$sample \
                        --filter "peakPicking true [1,2]" \
                        --filter "zeroSamples removeExtra" \
                        -o /data/mzMLs/
