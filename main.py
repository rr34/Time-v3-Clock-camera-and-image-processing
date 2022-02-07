import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import camera
from functions import *

cell_phone_1x_16to9 = camera.CameraAim()

# initialize this instance of the CameraAim class external to the __init__ function
cell_phone_1x_16to9.calibrate(calibration_file='slayer_mobile_calibrate_1xzoom_16to9aspectratio.csv')

xx = cell_phone_1x_16to9.pixel_aim[:,:,0]
yy = cell_phone_1x_16to9.pixel_aim[:,:,1]
zz = np.sqrt((cell_phone_1x_16to9.pixel_aim[:,:,2])**2 + (cell_phone_1x_16to9.pixel_aim[:,:,3])**2)

fig = pyplot.figure(figsize=[12,8])
ax = fig.gca(projection = '3d')
ax.plot_surface(xx, yy, zz, cmap = cm.copper)
pyplot.show()


"""
print(cell_phone_1x_16to9.pixel_aim[0::1000,0::1000,:])
print(cell_phone_1x.__dict__, '\n\n')
"""