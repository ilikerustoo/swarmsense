

# Swarmsense: Probabilistic Robotics Final Project

This project was forked from the original Crazyswarm2 repository and all setup steps should be taken from the official Crazyswarm2 documentation seen below.

For the `gesture_control.py` in the `crazyflie_examples` package, our code was based on this [hand gesture recognition](https://github.com/kinivi/hand-gesture-recognition-mediapipe) repo. This requires:


- mediapipe 0.8.11 (Original repo has incorrect version)
- OpenCV 3.4.2 or Later
- Tensorflow 2.8.0 (We fixed it 2.8.0 for shared dependency conflict reasons with mediapipe)
- tf-nightly 2.5.0.dev or later (Only when creating a TFLite for an LSTM model)
- scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix)
- matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix)

When running in a simulation or with cpp backend, run 
```
[terminal1]$ ros2 launch crazyflie launch.py backend:=sim
```
or 
```
[terminal1]$ ros2 launch crazyflie launch.py
```
and then
```
[terminal2]$ ros2 run crazyflie_examples control_gesture
```


[![ROS 2](https://github.com/IMRCLab/crazyswarm2/actions/workflows/ci-ros2.yml/badge.svg)](https://github.com/IMRCLab/crazyswarm2/actions/workflows/ci-ros2.yml)

# Crazyswarm2
A ROS 2-based stack for Bitcraze Crazyflie multirotor robots.

The documentation is available here: https://imrclab.github.io/crazyswarm2/.

## Troubleshooting
Please start a [Discussion](https://github.com/IMRCLab/crazyswarm2/discussions) for...

- Getting Crazyswarm2 to work with your hardware setup.
- Advice on how to use it to achieve your goals.
- Rough ideas for a new feature.

Please open an [Issue](https://github.com/IMRCLab/crazyswarm2/issues) if you believe that fixing your problem will involve a **change in the Crazyswarm2 source code**, rather than your own configuration files. For example...

- Bug reports.
- New feature proposals with details.


## SwarmSense Notes
 - Run and tested on Ubuntu 22.04 with ROS Iron
 - Please check and correct `crazyflie.yaml` to your crazyflie address setting and set each cf to true or false depending on how man you are connecting, otherwise connection will not work. E.g You are trying to connect one cf, but crazyswarm is expecting two cfs.
 - Install `nicegui` for simulation and visuals of drone in 3D space
 - Install the official Bitcraze client using `pip install cfclient` to set cf unique addresses or firmware upgreades
 - Only use the older crazyradio PA
 