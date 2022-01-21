import numpy as np
from matplotlib import pyplot
import camera
from functions import *

cell_phone_1x_16to9 = camera.CameraAim()

# initialize this instance of the CameraAim class external to the __init__ function
cell_phone_1x_16to9.calibrate(calibration_file='slayer_mobile_calibrate_1xzoom_16to9aspectratio.csv')

"""
print(cell_phone_1x.pixel_aim[:,:,2])

print(cell_phone_1x.__dict__, '\n\n')

print(cell_phone_1x.pixel_aim.shape, '\n\n')

print(cell_phone_1x.pixel_aim[0,0])

pyplot.imshow(cell_phone_1x.pixel_aim)
pyplot.show()
"""