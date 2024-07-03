#!/bin/bash
# use conda
# file: DIA_NN.pbs
#PBS -N DIA-NN
#PBS -l nodes=1:ppn=1
##PBS -l mem=10gb
#PBS -l walltime=72:00:00
#PBS -q default
#PBS -j oe
# import all environments virables
#PBS -V

software="/public/home/proteome/ranpeng/software"
mono_sif="/public/home/proteome/ranpeng/software/mono.sif"
ThermoRawFileParser="/public/home/proteome/ranpeng/software/ThermoRawFileParser1/ThermoRawFileParser.exe"

input_dir="/public/home/proteome/ranpeng/DataStorage/xinjiang_CTO/CTO_raw"
output_dir="/public/home/proteome/ranpeng/DataStorage/xinjiang_CTO/mzML_thermo"

if [ ! -d "$outout_dir" ]; then
    mkdir "$outout_dir"
    echo "Directocary created: $outout_dir"
else
    echo "Directory already exists: $outout_dir"
fi

# 判断文件是否已经存在
base_name=$(basename "$sample" .raw)
output_file="$output_dir/${base_name}.mzML"

if [ -f "$output_file" ]; then
    file_size=$(stat -c%s "$output_file")
    if [ $file_size -gt $((100 * 1024 * 1024)) ]; then
        echo "File already exists and is larger than 100MB: $output_file"
    else
        echo "File exists but is smaller than 100MB, reprocessing: $output_file"
        echo "ThermoRawFileParser for ${sample} is start running !"
        singularity exec --bind /public/group_share_data/proteome/ranpeng/:/public/group_share_data/proteome/ranpeng/ $mono_sif mono $ThermoRawFileParser -i $input_dir/$sample -o $outout_dir
        echo "ThermoRawFileParser for ${sample} is succesfully done !"
    fi
else
    echo "ThermoRawFileParser for ${sample} is start running!"
    singularity exec --bind /public/group_share_data/proteome/ranpeng/:/public/group_share_data/proteome/ranpeng/ $mono_sif mono $ThermoRawFileParser -i $input_dir/$sample -o $outout_dir
    echo "ThermoRawFileParser for ${sample} is successfully done!"
fi