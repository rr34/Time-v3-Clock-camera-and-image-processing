import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfile
import camera
from functions import *

# process a calibration .csv file, save as a (large) camera_aim .pkl file, visualize the calibration pixel/altaz data together with resulting pixel/altaz values
def generate_camera_aim_object():

    this_camera = camera.CameraAim()

    # initialize this instance of the CameraAim class external to the __init__ function
    Tk().withdraw()
    calibration_file = askopenfilename()
    this_camera.calibrate(calibration_file=calibration_file)

    save_to_file(this_camera, 'cell_phone_1x_16to9.pkl')

    xx = this_camera.pixel_aim[:,:,0]
    yy = this_camera.pixel_aim[:,:,1]
    zz_alt = this_camera.pixel_aim[:,:,2]
    zz_az = this_camera.pixel_aim[:,:,3]
    zz_az = this_camera.pixel_aim[:,:,3]
    xx_ref = this_camera.reference_pixels[:,0]
    yy_ref = this_camera.reference_pixels[:,1]
    zz_ref_alt = this_camera.reference_pixels[:,2]
    zz_ref_az = this_camera.reference_pixels[:,3]

    fig1, ax1 = pyplot.subplots(subplot_kw={"projection": "3d"})
    ax1.plot_surface(xx, yy, zz_alt, cmap=cm.inferno)
    ax1.scatter(xx_ref, yy_ref, zz_ref_alt, s=25, c='red')

    fig2, ax2 = pyplot.subplots(subplot_kw={"projection": "3d"})
    ax2.plot_surface(xx, yy, zz_az, cmap = cm.viridis)
    ax2.scatter(xx_ref, yy_ref, zz_ref_az, s=25, c='purple')

    pyplot.show()

    print(cell_phone_1x.__dict__, '\n\n')