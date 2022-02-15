import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import tkinter
import camera
from functions import *

def do_nothing():
    filewin = tkinter.Toplevel(root)
    button = Button(filewin, text='Do nothing button')
    Button.pack()

# process a calibration .csv file, save as a .awim_camera_aim file, visualize the calibration pixel/altaz data together with resulting pixel/altaz values
def generate_camera_aim_object():

    this_camera = camera.CameraAim()

    # initialize this instance of the CameraAim class external to the __init__ function
    calibration_file = tkinter.filedialog.askopenfilename()
    this_camera.calibrate(calibration_file=calibration_file)

    # camera_aim_destination_file = asksaveasfilename(defaultextension='.data')
    camera_aim_destination_file = str(this_camera.camera_name) + ' - lens ' + str(this_camera.lens_name) + ' - zoom ' + str(this_camera.zoom_factor) + '.awim_camera_aim'
    save_to_file(this_camera, camera_aim_destination_file)

    xx = this_camera.pixel_aim[:,:,0]
    yy = this_camera.pixel_aim[:,:,1]
    zz_alt = this_camera.pixel_aim[:,:,2]
    zz_az = this_camera.pixel_aim[:,:,3]
    zz_az = this_camera.pixel_aim[:,:,3]
    xx_ref = this_camera.reference_pixels[:,0]
    yy_ref = this_camera.reference_pixels[:,1]
    zz_ref_alt = this_camera.reference_pixels[:,2]
    zz_ref_az = this_camera.reference_pixels[:,3]

    fig, (ax1, ax2) = pyplot.subplots(1, 2, subplot_kw={"projection": "3d"})
    fig.suptitle('Alt / Az by Pixel')
    ax1.set_title('Altitude')
    ax1.plot_surface(xx, yy, zz_alt, cmap=cm.inferno)
    ax1.scatter(xx_ref, yy_ref, zz_ref_alt, s=25, c='red')
    ax2.set_title('Azimuth')
    ax2.plot_surface(xx, yy, zz_az, cmap = cm.viridis)
    ax2.scatter(xx_ref, yy_ref, zz_ref_az, s=25, c='purple')

    pyplot.show()

    print(cell_phone_1x.__dict__, '\n\n')