from tkinter.filedialog import askopenfilename
import os
import numpy as np
import math
import PIL
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import datetime
import pytz
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
import camera, awimlib, astropytools, XMPtext, formatters, metadata_tools


def generate_save_camera_AWIM():
    # create a CameraAWIMData object with calibration CSV file, then save using automatic name from calibration data
    calibration_image = askopenfilename(title='open calibration image')
    calibration_file = askopenfilename(title='open calibration csv file')
    camera_ID = camera.generate_camera_AWIM_from_calibration(calibration_image, calibration_file)
    

def display_camera_lens_shape(awim_dictionary):
    # open the object from a file, run its __repr__ method, use it to predict and plot its own predictions
    this_camera_filename = askopenfilename()
    camera_aim_pickle = open(this_camera_filename, 'rb')
    this_camera = pickle.load(camera_aim_pickle)
    camera_aim_pickle.close()

    this_camera.represent_camera()

    plot_dims = [160, 90]
    x_axis_px = np.linspace(0, this_camera.center_px[0], plot_dims[0])
    y_axis_px = np.linspace(0, this_camera.center_px[1], plot_dims[1])
    px_x_axis, px_y_axis = np.meshgrid(x_axis_px, y_axis_px)
    px_meshgrid = np.empty([plot_dims[1], plot_dims[0], 2])
    px_meshgrid[:,:,0] = px_x_axis
    px_meshgrid[:,:,1] = px_y_axis
    xyangs_predicted = this_camera.px_xyangs_models_convert(input=px_meshgrid, direction='px_to_xyangs')

    fig1, (ax1, ax2) = pyplot.subplots(1, 2, subplot_kw={"projection": "3d"})
    fig1.suptitle('x,y Angles from Pixels')
    ax1.set_title('x Angle')
    ax1.plot_surface(px_meshgrid[:,:,0], px_meshgrid[:,:,1], xyangs_predicted[:,:,0], cmap = cm.viridis)
    ax1.scatter(this_camera.ref_df['x_px'], this_camera.ref_df['y_px'], this_camera.ref_df['xang'], s=50, c='purple')
    ax2.set_title('y Angle')
    ax2.plot_surface(px_meshgrid[:,:,0], px_meshgrid[:,:,1], xyangs_predicted[:,:,1], cmap=cm.inferno)
    ax2.scatter(this_camera.ref_df['x_px'], this_camera.ref_df['y_px'], this_camera.ref_df['yang'], s=50, c='red')

    xang_axis = np.linspace(this_camera.xyangs_edges[3,0], this_camera.xyangs_edges[1,0], plot_dims[0])
    yang_axis = np.linspace(this_camera.xyangs_edges[2,1], this_camera.xyangs_edges[0,1], plot_dims[1])
    xang_axis, yang_axis = np.meshgrid(xang_axis, yang_axis)
    xyangs_meshgrid = np.empty([plot_dims[1], plot_dims[0], 2])
    xyangs_meshgrid[:,:,0] = xang_axis
    xyangs_meshgrid[:,:,1] = yang_axis
    px_predicted = this_camera.px_xyangs_models_convert(input=xyangs_meshgrid, direction='xyangs_to_px')

    fig2, (ax3, ax4) = pyplot.subplots(1, 2, subplot_kw={"projection": "3d"})
    fig2.suptitle('Pixels from x,y Angles')
    ax3.set_title('x Pixel')
    ax3.plot_surface(xyangs_meshgrid[:,:,0], xyangs_meshgrid[:,:,1], px_predicted[:,:,0], cmap = cm.viridis)
    ax3.scatter(this_camera.ref_df['xang'], this_camera.ref_df['yang'], this_camera.ref_df['x_px'], s=50, c='purple')
    ax4.set_title('y Pixel')
    ax4.plot_surface(xyangs_meshgrid[:,:,0], xyangs_meshgrid[:,:,1], px_predicted[:,:,1], cmap=cm.inferno)
    ax4.scatter(this_camera.ref_df['xang'], this_camera.ref_df['yang'], this_camera.ref_df['y_px'], s=50, c='red')

    pyplot.show()


def do_nothing():
    pass


def lightroom_timelapse_XMP_process():
# self.bind('<Control-Key-d>', self.select_XMPdirectory)
# self.bind('<Control-Key-1>', self.step1_initial_label_XMPs)
# self.bind('<Control-Key-2>', self.step2_interpolate)

    XMPdirectory = os.path.join(os.getcwd(), 'working')

    # read XMP files
    columns_to_interpolate = ['crs Temperature', 'crs Tint', 'crs Exposure2012', 'crs Contrast2012', 'crs Highlights2012', 'crs Shadows2012', 'crs Whites2012', 'crs Blacks2012', 'crs Texture', 'crs Clarity2012', 'crs Dehaze', 'crs Vibrance', 'crs Saturation']
    XMP_snapshot, lapse_latlng = XMPtext.readXMPfiles(XMPdirectory, columns_to_interpolate)
    XMP2 = XMP_snapshot.copy()
# set variables for sun and moon calculations
    moments_list = XMP2['exif DateTimeOriginal'].values
    moments_list = formatters.format_datetimes(input_datetime=moments_list, direction='from list of ISO 8601 strings')
    print(lapse_latlng)
# calculate sun and moon values
    sun_az_list, sun_art_list = astropytools.get_AzArts(earth_latlng=lapse_latlng, moments=moments_list, celestial_object='sun')
    moon_az_list, moon_art_list = astropytools.get_AzArts(earth_latlng=lapse_latlng, moments=moments_list, celestial_object='moon')
# convert sun and moon values to day, night, twilight labels, format numbers, add to dataframe
    day_night_twilight_list = astropytools.day_night_twilight(sun_art_list, moon_art_list)
    sun_az_list = formatters.round_to_string(sun_az_list, 'azimuth')
    sun_art_list = formatters.round_to_string(sun_art_list, 'artifae')
    moon_az_list = formatters.round_to_string(moon_az_list, 'azimuth')
    moon_art_list = formatters.round_to_string(moon_art_list, 'artifae')
    XMP2['awim SunAz'] = sun_az_list
    XMP2['awim SunArt'] = sun_art_list
    XMP2['awim MoonAz'] = moon_az_list
    XMP2['awim MoonArt'] = moon_art_list
    XMP2['awim DayNightTwilight'] = day_night_twilight_list
# concatenate new tags together with the old tags, comma-separated
    XMP2['awim CommaSeparatedTags'] = XMP2.apply(lambda x:'%s,%s' % (x['awim CommaSeparatedTags'], x['awim DayNightTwilight']), axis=1)
# save dataframe to CSV file
    timenow = datetime.datetime.now()
    time_string = formatters.format_datetimes(timenow, 'to string for filename')
    filename = f'XMP_step1 {time_string}.csv'
    filepath = os.path.join(XMPdirectory, filename)
    XMP2.to_csv(filepath)

# write the comma-separated tags to the XMP files
    XMPtext.addTags(XMP_snapshot, XMP2, XMPdirectory)
    print('Completed step 1 labelling XMP files with cellestial events.')

    XMP_snapshot, lapse_latlng = XMPtext.readXMPfiles(XMPdirectory, columns_to_interpolate)
    XMP2 = XMP_snapshot.copy() # this seems unnecessary since XMP2 is defined in the next line, but maybe reauired to prevent XMP2 from pointing to XMP_snapshot.
    print('Interpolating the dataframe of XMP values...')
    XMP2 = XMPtext.interpolate(XMP_snapshot, columns_to_interpolate)
# save dataframe to CSV file
    print('Saving interpolated dataframe to CSV...')
    timenow = datetime.datetime.now()
    time_string = formatters.format_datetimes(timenow, 'to string for filename')
    filename = f'XMP_step2 {time_string}.csv'
    filepath = os.path.join(XMPdirectory, filename)
    XMP2.to_csv(filepath)

# write the new values to the XMP files
    XMPtext.write_values(XMP2, columns_to_interpolate, XMPdirectory)
    print('Completed step 2 interpolating between the keyframes and writing to XMP files.')


def generate_metatext_files():
    workingpath = os.path.join(os.getcwd(), 'working')
    metadata_tools.meta_to_textfiles(os.path.join(workingpath))