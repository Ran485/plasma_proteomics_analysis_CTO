#!/bin/bash
# use conda
#PBS -N DIA-NN
#PBS -l nodes=1:ppn=1
##PBS -l mem=10gb
#PBS -l walltime=72:00:00
#PBS -q default
#PBS -j oe
#PBS -V

# Change to the directory where the program is located
cd /public/home/proteome/ranpeng/DataStorage/xinjiang_CTO

# 配置 'ThermoRawFileParser' 的路径，一般不用更改
software="/public/home/proteome/ranpeng/software"
mono_sif="/public/home/proteome/ranpeng/software/mono.sif"
ThermoRawFileParser="/public/home/proteome/ranpeng/software/ThermoRawFileParser1/ThermoRawFileParser.exe"

# 配置 'RAW data' 输入及其 'mzML' 文件的输出目录
input_dir="/public/home/proteome/ranpeng/DataStorage/xinjiang_CTO/CTO_raw"
output_dir="/public/home/proteome/ranpeng/DataStorage/xinjiang_CTO/mzML_thermo"

if [ ! -d "$output_dir" ]; then
    mkdir "$output_dir"
    echo "Directory created: $output_dir"
else
    echo "Directory already exists: $output_dir"
fi

# 配置 DIANN 搜库的参数，一般不用更改
FILE_TYPE=".mzML"
FILE_PATH="$output_dir"
LIB_PATH="/public/home/proteome/ranpeng/DIANN/library/"
THREADS="1"
DIA_NN="/public/software/apps/singularity/3.5.2/bin/singularity exec --bind /public/group_share_data/proteome/ranpeng/:/public/group_share_data/proteome/ranpeng/ ubuntu_rp.sif /usr/diann/1.8.1/diann-1.8.1"

# 配置 DIANN 结果输出目录
OUT_PATH="./Results/"
TMP_PATH="./tmp/"

if [ ! -d "$OUT_PATH" ]; then
    mkdir -p "$OUT_PATH"
fi

if [ ! -d "$TMP_PATH" ]; then
    mkdir -p "$TMP_PATH"
fi


# 1. Step 1: ThermoRawFileParser for mzML conversion
base_name=$(basename "$sample" .raw)
output_file="$output_dir/${base_name}.mzML"

if [ -f "$output_file" ]; then
    file_size=$(stat -c%s "$output_file")
    if [ $file_size -gt $((100 * 1024 * 1024)) ]; then
        echo "File already exists and is larger than 100MB: $output_file"
    else
        echo "File exists but is smaller than 100MB, reprocessing: $output_file"
        echo "ThermoRawFileParser for ${sample} is start running!"
        singularity exec --bind /public/group_share_data/proteome/ranpeng/:/public/group_share_data/proteome/ranpeng/ $mono_sif mono $ThermoRawFileParser -i $input_dir/$sample -o $output_dir
        echo "ThermoRawFileParser for ${sample} is successfully done!"
    fi
else
    echo "ThermoRawFileParser for ${sample} is start running!"
    singularity exec --bind /public/group_share_data/proteome/ranpeng/:/public/group_share_data/proteome/ranpeng/ $mono_sif mono $ThermoRawFileParser -i $input_dir/$sample -o $output_dir
    echo "ThermoRawFileParser for ${sample} is successfully done!"
fi

# Step 2: run DIANN pipeline
python3 diann_327_FDR001.py "$base_name.mzML" "$FILE_PATH" "$LIB_PATH" "$OUT_PATH" "$TMP_PATH" "$THREADS" "$DIA_NN"