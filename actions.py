import numpy as np
import tkinter
import pickle
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import camera
from functions import *

def do_nothing():
    filewin = tkinter.Toplevel(root)
    button = Button(filewin, text='Do nothing button')
    Button.pack()

# process a calibration .csv file, save as a .awim, visualize the calibration pixel/azalt data together with resulting pixel/azalt values
def generate_camera_aim_object():

    # create a CameraAim object with calibration CSV file, then save using automatic name from calibration data
    calibration_file = tkinter.filedialog.askopenfilename()
    this_camera = camera.CameraAim(calibration_file)

    # save to a pickle .awim_camera_aim_pkl
    this_camera_filename = str(this_camera.camera_name) + ' - lens ' + str(this_camera.lens_name) + ' - zoom ' + str(this_camera.zoom_factor) + '.awim'
    camera_aim_pickle = open(this_camera_filename, 'wb')
    pickle.dump(this_camera, camera_aim_pickle, 5)
    camera_aim_pickle.close()

def display_camera_aim_object():
    # open the object from a file, run its __repr__ method, use it to predict and plot its own predictions
    this_camera_filename = tkinter.filedialog.askopenfilename()
    camera_aim_pickle = open(this_camera_filename, 'rb')
    this_camera = pickle.load(camera_aim_pickle)
    camera_aim_pickle.close()

    # this_camera.represent_camera()

    plot_dims = [160, 90]
    x_axis_px = np.linspace(0, this_camera.center_px[0], plot_dims[0])
    y_axis_px = np.linspace(0, this_camera.center_px[1], plot_dims[1])
    px_x_axis, px_y_axis = np.meshgrid(x_axis_px, y_axis_px)
    px_meshgrid = np.empty([plot_dims[1], plot_dims[0], 2])
    px_meshgrid[:,:,0] = px_x_axis
    px_meshgrid[:,:,1] = px_y_axis

    azalt_predicted = this_camera.cam_px_to_azalt(px=px_meshgrid)

    fig, (ax1, ax2) = pyplot.subplots(1, 2, subplot_kw={"projection": "3d"})
    fig.suptitle('Az / Alt from Pixels')
    ax1.set_title('Azimuth')
    ax1.plot_surface(px_meshgrid[:,:,0], px_meshgrid[:,:,1], azalt_predicted[:,:,0], cmap = cm.viridis)
    ax1.scatter(this_camera.reference_pixels['x_px'], this_camera.reference_pixels['y_px'], this_camera.reference_pixels['az'], s=50, c='purple')
    ax2.set_title('Altitude')
    ax2.plot_surface(px_meshgrid[:,:,0], px_meshgrid[:,:,1], azalt_predicted[:,:,1], cmap=cm.inferno)
    ax2.scatter(this_camera.reference_pixels['x_px'], this_camera.reference_pixels['y_px'], this_camera.reference_pixels['alt'], s=50, c='red')

    pyplot.show()