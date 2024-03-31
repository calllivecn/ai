#!/bin/bash
# date 2023-05-06 09:07:17
# author calllivecn <calllivecn@outlook.com>

# 需要 先关闭所有在使用cuda的进程。可以使用nvidia-smi 来查看。

# rmmod: ERROR: Module nvidia_drm is in use
# rmmod: ERROR: Module nvidia_modeset is in use by: nvidia_drm
# rmmod: ERROR: Module nvidia_uvm is in use
# rmmod: ERROR: Module nvidia is in use by: nvidia_uvm nvidia_modeset

rmmod nvidia_drm
rmmod nvidia_modeset
rmmod nvidia_uvm
rmmod nvidia

modprobe nvidia
modprobe nvidia_uvm
modprobe nvidia_modeset
modprobe nvidia_drm

