#!/usr/bin/env bash
# Usage:
# bash scripts/user_tools/setup_processed_bag_directory.sh <number of bags> <algorithm name> <platform> <dataset name (lower case)> <dataset name (upper case)>
# eg:
#   bash scripts/user_tools/setup_processed_bag_directory.sh 9 rovio arm mh_01 MH_01
#
# Must be ran from the rpg_trajectory_evaluation directory

total_bags=$1
alg=$2
arch=$3
bag_datasets_name=$4
updated_bag_datasets_name=$5

result_path="/root/catkin_ws/src/rpg_trajectory_evaluation/results/av_euroc_vio_mono/"$arch"/"$alg"/"$arch"_"$alg"_"$updated_bag_datasets_name""
echo result_path = $result_path

echo $total_bags

for ((i = 0 ; i <= $total_bags ; i++)); do
  echo "Making Directory for $alg , Test $i"

  cd "$result_path"
  mkdir -p "$alg"-test-"$i"

done
