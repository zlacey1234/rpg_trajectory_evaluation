#!/usr/bin/env bash
# Usage:
# bash scripts/user_tools/process_bags.sh <number of bags> <algorithm name> <platform> <dataset name (lower case)> <dataset name (upper case)>
# eg:
#   bash scripts/user_tools/process_bags.sh 9 rovio arm mh_01 MH_01
#
# Must be ran from the rpg_trajectory_evaluation directory

total_bags=$1
alg=$2
arch=$3
bag_datasets_name=$4
updated_bag_datasets_name=$5

result_path="/root/catkin_ws/src/rpg_trajectory_evaluation/results/av_euroc_vio_mono/"$arch"/"$alg"/"$arch"_"$alg"_"$updated_bag_datasets_name""
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

echo $total_bags

for ((i = 0 ; i <= $total_bags ; i++)); do
  echo "PROCESSING BAG $i"

  python scripts/dataset_tools/bag_to_pose.py "$result_path"/"$alg"_traj_"$i".bag $topic_name --msg_type=$topic_type

  mv "$result_path"/stamped_poses.txt "$result_path"/"$alg"-test-"$i"/stamped_traj_estimate.txt
  cp "$result_path"/eval_cfg.yaml "$result_path"/"$alg"-test-"$i"/
  cp "$result_path"/stamped_groundtruth.txt "$result_path"/"$alg"-test-"$i"/

done
