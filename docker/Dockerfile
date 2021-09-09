FROM ros:melodic-perception

ENV CATKIN_WS=/root/catkin_ws
WORKDIR ${CATKIN_WS}/src

RUN sudo apt-get update && apt-get install -y \
	ros-${ROS_DISTRO}-rviz \
	python-catkin-tools \
	python3-pip \
	python3-matplotlib \
	texlive-fonts-recommended texlive-fonts-extra dvipng \
	python3-tk \
	libnotify-bin \
	vim


RUN pip3 install -U PyYAML && \
	pip3 install numpy && \
	pip3 install colorama && \
	pip3 install setuptools wheel && \
 	pip3 install ruamel.yaml && \
 	pip3 install pandas && \
    pip3 install pycryptodomex && \
    pip3 install gnupg

# USE BASH
SHELL ["/bin/bash", "-c"]

# Create the Catkin Workspace (Note: bootstrap rosdep is done in the base
# image)
RUN echo ". $CATKIN_WS/devel/setup.bash" >> ~/.bashrc \
    && source /opt/ros/${ROS_DISTRO}/setup.sh \
    && mkdir -p ~/catkin_ws/src/ \
    && cd ~/catkin_ws && catkin init && catkin build \
    && source devel/setup.bash

RUN cd ~/catkin_ws/src \
    && git clone https://github.com/zlacey1234/rpg_trajectory_evaluation.git \
    && git clone https://github.com/catkin/catkin_simple.git

RUN cd ~/catkin_ws/ \
    && catkin build

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc \
    && echo "source $CATKIN_WS/devel/setup.bash" >> ~/.bashrc

ADD   ros-entrypoint.sh /ros-entrypoint.sh
RUN   sudo chmod +X /ros-entrypoint.sh
ENTRYPOINT [ "/ros-entrypoint.sh" ]
