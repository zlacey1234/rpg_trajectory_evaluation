# enable access to xhost from the container
xhost +

# Run docker and open bash shell
docker run -it --privileged --rm \
--env=LOCAL_USER_ID="$(id -u)" \
--env QT_X11_NO_MITSHM=1 \
--user root \
-v ~/project/rpg_trajectory_evaluation:/root/catkin_ws/src/rpg_trajectory_evaluation \
-v ~/projects/vio_datasets:/root/vio_datasets \
-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
-v /dev:/dev:ro \
-e DISPLAY=$DISPLAY \
--name=vio_traj_eval ros:vio_traj_eval bash
