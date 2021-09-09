tag=ros:vio_traj_eval

docker build \
  --network=host \
  -t "$tag" \
  -f Dockerfile \
  .
