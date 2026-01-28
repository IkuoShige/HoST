#!/bin/bash
export DISPLAY=:1
xhost +
docker run --rm -it --gpus all --runtime=nvidia \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e "ACCEPT_EULA=Y" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --name host \
  host:latest

  #--entrypoint bash \
