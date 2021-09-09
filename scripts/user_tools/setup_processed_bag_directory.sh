#!/usr/bin/env bash
# Usage:
# bash scripts/user_tools/setup_processed_bag_directory.sh <test number> <algorithm name> <platform>
#   <dataset name (upper case)> <output directory>
# eg:
#   bash scripts/user_tools/setup_processed_bag_directory.sh 1 rovio arm MH_01 ./results/av_euroc_vio_mono
#
# Must be ran from the rpg_trajectory_evaluation directory
test_num=$1
alg=$2
arch=$3
updated_bag_datasets_name=$4
output_dir=$5

result_path=$output_dir"/"$arch"/"$alg"/"$arch"_"$alg"_"$updated_bag_datasets_name""

echo result_path = $result_path

echo $test_num

echo "Making Directory for $alg , Test $test_num"

cd "$result_path"
mkdir -p "$alg"-test-"$test_num"
