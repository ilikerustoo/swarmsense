from crazyflie_py import Crazyswarm
import numpy as np
import time

def main():
    # X = -1.0

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs
    # allcf = swarm.allcfs.crazyflies[0]
    allcfs.takeoff(targetHeight=1.0, duration=1.0)
    # timeHelper.sleep(1.5)
    
    # for cf in allcfs.crazyflies:
    #     pos = np.array(cf.initialPosition) + np.array([X, 0, 0])
    #     cf.goTo(pos, 0, 1.0)
    # print('press button to continue...')
    # swarm.input.waitUntilButtonPressed()

    # # allcfs.land(targetHeight=0.02, duration=1.0+Z)
    # timeHelper.sleep(1.0)


if __name__ == '__main__':
    main()
