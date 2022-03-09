import numpy as np
import tkinter
import pickle
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import PIL
import datetime
from pytz import timezone
import pytz
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

    this_camera.represent_camera()

    plot_dims = [160, 90]
    x_axis_px = np.linspace(0, this_camera.center_px[0], plot_dims[0])
    y_axis_px = np.linspace(0, this_camera.center_px[1], plot_dims[1])
    px_x_axis, px_y_axis = np.meshgrid(x_axis_px, y_axis_px)
    px_meshgrid = np.empty([plot_dims[1], plot_dims[0], 2])
    px_meshgrid[:,:,0] = px_x_axis
    px_meshgrid[:,:,1] = px_y_axis
    azalt_predicted = this_camera.px_azalt_models_convert(input=px_meshgrid, direction='px_to_azalt')

    fig1, (ax1, ax2) = pyplot.subplots(1, 2, subplot_kw={"projection": "3d"})
    fig1.suptitle('Az / Alt from Pixels')
    ax1.set_title('Azimuth')
    ax1.plot_surface(px_meshgrid[:,:,0], px_meshgrid[:,:,1], azalt_predicted[:,:,0], cmap = cm.viridis)
    ax1.scatter(this_camera.ref_df['px_x'], this_camera.ref_df['px_y'], this_camera.ref_df['az'], s=50, c='purple')
    ax2.set_title('Altitude')
    ax2.plot_surface(px_meshgrid[:,:,0], px_meshgrid[:,:,1], azalt_predicted[:,:,1], cmap=cm.inferno)
    ax2.scatter(this_camera.ref_df['px_x'], this_camera.ref_df['px_y'], this_camera.ref_df['alt'], s=50, c='red')

    az_axis = np.linspace(this_camera.azalt_edges[3,0], this_camera.azalt_edges[1,0], plot_dims[0])
    alt_axis = np.linspace(this_camera.azalt_edges[2,1], this_camera.azalt_edges[0,1], plot_dims[1])
    az_axis, alt_axis = np.meshgrid(az_axis, alt_axis)
    azalt_meshgrid = np.empty([plot_dims[1], plot_dims[0], 2])
    azalt_meshgrid[:,:,0] = az_axis
    azalt_meshgrid[:,:,1] = alt_axis
    px_predicted = this_camera.px_azalt_models_convert(input=azalt_meshgrid, direction='azalt_to_px')

    fig2, (ax3, ax4) = pyplot.subplots(1, 2, subplot_kw={"projection": "3d"})
    fig2.suptitle('Pixels from Az / Alt')
    ax3.set_title('x Pixel')
    ax3.plot_surface(azalt_meshgrid[:,:,0], azalt_meshgrid[:,:,1], px_predicted[:,:,0], cmap = cm.viridis)
    ax3.scatter(this_camera.ref_df['az'], this_camera.ref_df['alt'], this_camera.ref_df['px_x'], s=50, c='purple')
    ax4.set_title('y Pixel')
    ax4.plot_surface(azalt_meshgrid[:,:,0], azalt_meshgrid[:,:,1], px_predicted[:,:,1], cmap=cm.inferno)
    ax4.scatter(this_camera.ref_df['az'], this_camera.ref_df['alt'], this_camera.ref_df['px_y'], s=50, c='red')

    pyplot.show()

def get_exif(current_image):
    from PIL.ExifTags import TAGS, GPSTAGS

    img_exif = current_image._getexif()

    img_exif_readable = {}
    for key in img_exif.keys():
        decode = TAGS.get(key,key)
        img_exif_readable[decode] = img_exif[key]

    exif_date, exif_time = (img_exif_readable['DateTime']).split(' ')[0], (img_exif_readable['DateTime']).split(' ')[1]
    year = int(exif_date.split(':')[0])
    month = int(exif_date.split(':')[1])
    day = int(exif_date.split(':')[2])
    hour = int(exif_time.split(':')[0])
    minute = int(exif_time.split(':')[1])
    second = int(exif_time.split(':')[2])

    # pytz info input
    time_zone_source = 'pytz'
    this_timezone_str = 'US/Eastern' # pytz can handle this Time v2 trash. Coorects for DST but not well at the switch. Do manually near switch.
    
    # manual offset input
    time_zone_source = 'manual offset'
    time_zone_offset_hrs = -5
    time_zone_offset_seconds = time_zone_offset_hrs*3600

    if time_zone_source == 'pytz':
        this_timezone = timezone(this_timezone_str) # creates a pytz class to convert datetimes
        image_moment_pytz = this_timezone.localize(datetime.datetime(year, month, day, hour, minute, second))
        time_zone_offset_seconds = image_moment_pytz.utcoffset().total_seconds()
    
    tzinfo = datetime.timezone(datetime.timedelta(seconds=time_zone_offset_seconds))
    img_capture_moment = datetime.datetime(year, month, day, hour, minute, second, 0, tzinfo)
    img_capture_moment = img_capture_moment.astimezone(datetime.timezone.utc)

    if img_exif.get(34853): # if GPS info present. Returns None if not.
        GPS_info_present = True
        exif_gps = {}
        for key in img_exif[34853].keys():
            decode = GPSTAGS.get(key,key)
            exif_gps[decode] = img_exif[34853][key]

        if exif_gps['GPSLatitudeRef'] == 'N':
            lat_sign = 1
        elif exif_gps['GPSLatitudeRef'] == 'S':
            lat_sign = -1
        else:
            print('no GPS North / South information')
        img_lat = lat_sign*float(exif_gps['GPSLatitude'][0] + exif_gps['GPSLatitude'][1]/60 + exif_gps['GPSLatitude'][2]/3600)
        if exif_gps['GPSLongitudeRef'] == 'E':
            lng_sign = 1
        elif exif_gps['GPSLongitudeRef'] == 'W':
            lng_sign = -1
        else:
            print('no GPS East / West information')

        img_lng = lng_sign*float(exif_gps['GPSLongitude'][0] + exif_gps['GPSLongitude'][1]/60 + exif_gps['GPSLongitude'][2]/3600)
        img_latlng = [img_lat, img_lng]

        elevRef = exif_gps['GPSAltitudeRef']
        elevRef = elevRef.decode('utf-8') # TODO: get the elevation sign from the byte-encoded Ref
        if elevRef == '\x00':
            elevRef_sign = 1
        else:
            print('Elevation is something unusual, probably less than zero like in Death Valley or something. Look here.')
            elevRef_sign = -1
        img_elevation = elevRef_sign * float(exif_gps['GPSAltitude'])
    else:
        GPS_info_present = False
        img_latlng = [9999.0, 9999.0]
        img_elevation = 9999.0

    return GPS_info_present, img_latlng, img_elevation, img_capture_moment

def generate_png_with_awim(exif_dictionary, destination_filename):
		# create the info object, add the awim data to the info object, save the png with the info object 
		png_data_container = PIL.PngImagePlugin.PngInfo()
		for key, value in exif_dictionary.items():
			png_data_container.add_text(key, value)

		current_image.save('slayer_cal_image_with_bigtxtdictionary2.png', 'PNG', pnginfo=png_data_container)



def png_text_reader(image_filename):
    png_file_1 = PIL.Image.open('slayer_cal_image_with_bigtxtdictionary2.png')


    """
    camera_data_retiever = PngImagePlugin.PngInfo()

    camera_data_from_png = camera_data_retiever.read(png_file_1)
    """

    png_text_dictionary = png_file_1.text

    print(png_text_dictionary['Az / Alt Model'])