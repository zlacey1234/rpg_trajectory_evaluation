#!/usr/bin/env bash
# Usage:
# bash scripts/user_tools/process_bags.sh <test number> <algorithm name> <platform>
#   <dataset name (upper case)> <output directory>
# eg:
#   bash scripts/user_tools/process_bags.sh 1 rovio arm MH_01 ./results/av_euroc_vio_mono
#
# Must be ran from the rpg_trajectory_evaluation directory
test_num=$1
alg=$2
arch=$3
updated_bag_datasets_name=$4
output_dir=$5

result_path=$output_dir"/"$arch"/"$alg"/"$arch"_"$alg"_"$updated_bag_datasets_name""

echo result_path = $result_path
topic_name="init"
topic_type="init"

if [ "$alg" = "rovio" ]; then
  echo "processing rovio bags"
  topic_name="/rovio/odometry"
  topic_type="PoseWithCovarianceStamped"
elif [ "$alg" = "larvio" ]; then
  echo "processing larvio bags"
  topic_name="/qxc_robot/system/odom"
  topic_type="PoseWithCovarianceStamped"
elif [ "$alg" = "svo2" ]; then
  echo "processing svo2 bags"
  topic_name="/svo/pose_imu"
  topic_type="PoseWithCovarianceStamped"
else
  echo "unknown algorithm"
  exit 1
fi

echo "PROCESSING BAG $test_num"

python scripts/dataset_tools/bag_to_pose.py "$result_path"/"$alg"_traj_"$test_num".bag $topic_name --msg_type=$topic_type

mv "$result_path"/stamped_poses.txt "$result_path"/"$alg"-test-"$test_num"/stamped_traj_estimate.txt
cp "$result_path"/eval_cfg.yaml "$result_path"/"$alg"-test-"$test_num"/
cp "$result_path"/stamped_groundtruth.txt "$result_path"/"$alg"-test-"$test_num"/

