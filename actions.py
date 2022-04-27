import tkinter
import pickle
import PIL
import os
import numpy as np
import math
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import datetime
from pytz import timezone
import pytz
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
import camera


def do_nothing():
    filewin = tkinter.Toplevel(root)
    button = Button(filewin, text='Do nothing button')
    Button.pack()


def generate_save_camera_AWIM():

    # create a CameraAWIMData object with calibration CSV file, then save using automatic name from calibration data
    calibration_file = tkinter.filedialog.askopenfilename()
    this_camera = camera.CameraAWIMData(calibration_file)

    # save to a pickle .awim_camera_aim_pkl
    this_camera_filename = str(this_camera.camera_name) + ' - lens ' + str(this_camera.lens_name) + ' - zoom ' + str(this_camera.zoom_factor) + '.awim'
    camera_aim_pickle = open(this_camera_filename, 'wb')
    pickle.dump(this_camera, camera_aim_pickle, 5)
    camera_aim_pickle.close()


def display_camera_AWIM_object():
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


def get_exif(current_image):
    from PIL.ExifTags import TAGS, GPSTAGS
    exif_present = False
    GPS_info_present = False

    rotate_degrees = 0
    img_latlng = None
    img_elevation = None
    img_capture_moment = None
    time_offset_hrs = None

    img_exif = current_image._getexif()

    if img_exif:
        exif_present = True
        img_exif_readable = {}
        for key in img_exif.keys():
            decode = TAGS.get(key,key)
            img_exif_readable[decode] = img_exif[key]

        # check rotation.
        if not math.isnan(img_exif_readable['Orientation']):
            if img_exif_readable['Orientation'] == 1:
                rotate_degrees = 0
            elif img_exif_readable['Orientation'] == 3:
                rotate_degrees = 180
            elif img_exif_readable['Orientation'] == 6:
                rotate_degrees = 90
            elif img_exif_readable['Orientation'] == 8:
                rotate_degrees = 270

        exif_date, exif_time = (img_exif_readable['DateTimeOriginal']).split(' ')[0], (img_exif_readable['DateTimeOriginal']).split(' ')[1]
        year = int(exif_date.split(':')[0])
        month = int(exif_date.split(':')[1])
        day = int(exif_date.split(':')[2])
        hour = int(exif_time.split(':')[0])
        minute = int(exif_time.split(':')[1])
        second = int(exif_time.split(':')[2])

        # TODO make this an input rather than hard-code set here. Goal is to enter total offset from UTC and have it converted to UTC
        # get the pytz timezone offset on the date of the photo, then include other time offset:
        # camera time in exif compared to local time including DST on the date of the photo
        this_timezone_str = 'US/Eastern' # pytz can handle this Time v2 stuff. Corrects for DST but not well at the switch. Do manually near switch.
        this_timezone_pytz = timezone(this_timezone_str) # creates a pytz class to convert datetimes
        img_moment_pytz = this_timezone_pytz.localize(datetime.datetime(year, month, day, hour, minute, second))
        img_pytz_offset_seconds = img_moment_pytz.utcoffset().total_seconds()

        offset_direction = -1
        offset_hrs = 0
        offset_minutes = 0
        offset_seconds = 0
        img_other_offset_seconds = offset_direction * (offset_hrs*3600 + offset_minutes*60 + offset_seconds)
        img_total_time_offset_seconds = img_pytz_offset_seconds + img_other_offset_seconds

        img_total_time_offset_seconds_astimezone = datetime.timezone(datetime.timedelta(seconds=img_total_time_offset_seconds))
        time_offset_hrs = img_time_offset_seconds/3600

        img_capture_moment = datetime.datetime(year, month, day, hour, minute, second, 0, img_total_time_offset_seconds_astimezone)
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
            img_latlng = [9999.0, 9999.0]
            img_elevation = 9999.0

    return rotate_degrees, exif_present, GPS_info_present, img_latlng, img_elevation, img_capture_moment, time_offset_hrs


def generate_png_with_awim_tag(current_image, rotate_degrees, awim_dictionary):
	# create the info object, add the awim data to the info object, save the png with the info object 
    png_data_container = PIL.PngImagePlugin.PngInfo()

    for key, value in awim_dictionary.items():
        png_data_container.add_text(key, value)
    
    save_filename_string = os.path.splitext(current_image.filename)[0] + ' - awim.png'
    current_image = current_image.rotate(angle=rotate_degrees, expand=True) # rotates CW
    current_image.save(save_filename_string, 'PNG', pnginfo=png_data_container)


def png_text_reader(image_filename):
    png_file_1 = PIL.Image.open(image_filename)
    png_text_dictionary = png_file_1.text

    for key, value in png_text_dictionary.items():
        print(key, ':', png_text_dictionary[key])

    return png_text_dictionary


def azalt_ref_from_known_px(camera, image, capture_moment, earth_latlng, center_ref, known_azalt_in, known_pt_px, img_orientation, img_tilt):
    image_dimensions = image.size
    max_image_index = np.subtract(image_dimensions, 1)
    img_center = np.divide(max_image_index, 2)
    if center_ref == 'center':
        center_ref = img_center
    img_aspect_ratio = image_dimensions[0] / image_dimensions[1]
    cam_aspect_ratio = camera.cam_image_dimensions[0] / camera.cam_image_dimensions[1]
    if abs(img_aspect_ratio - cam_aspect_ratio) > .001:
        print('error: image aspect ratio does not match camera aspect ratio, but it should')
    else:
        img_resize_factor = image_dimensions[0] / camera.cam_image_dimensions[0]

    known_pt_px_rel = [known_pt_px[0] - center_ref[0], center_ref[1] - known_pt_px[1]]

    img_astropy_time = Time(capture_moment)
    img_astropy_location = EarthLocation(lat=earth_latlng[0]*u.deg, lon=earth_latlng[1]*u.deg)
    img_astropy_frame = AltAz(obstime=img_astropy_time, location=img_astropy_location)
    if known_azalt_in == 'sun':
        object_altaz = get_sun(img_astropy_time).transform_to(img_astropy_frame)
        known_azalt = [object_altaz.az.degree, object_altaz.alt.degree]
    elif isinstance(known_azalt_in, (list, np.ndarray)):
        known_azalt = known_azalt_in
    object_xyangs_relcam = camera.px_xyangs_models_convert(input=np.divide(known_pt_px_rel, img_resize_factor), direction='px_to_xyangs')

    # variables names are from diagram included in the notes:
    if object_xyangs_relcam[0] < -90 or object_xyangs_relcam[0] > 90:
        print ('celestial object must be in front of camera, not super far off to the side.')
    xang_rel_rad = object_xyangs_relcam[0] * math.pi/180
    yang_rel_rad = object_xyangs_relcam[1] * math.pi/180
    x_direction = math.copysign(1, xang_rel_rad)
    y_direction = math.copysign(1, yang_rel_rad)
    xang_rel_rad = abs(xang_rel_rad)
    yang_rel_rad = abs(yang_rel_rad)
    obj_alt_rad = known_azalt[1] * math.pi/180
    d1 = 1*math.cos(math.pi/2 - xang_rel_rad)
    r2 = 1*math.sin(math.pi/2 - xang_rel_rad)
    alt_seg = 1*math.sin(obj_alt_rad)
    alt_ref_rad = math.asin(alt_seg / r2) - y_direction*yang_rel_rad
    d2 = r2 * math.cos(alt_ref_rad + yang_rel_rad)
    az_rel_rad = math.pi/2 - math.atan(d2 / d1)

    az_rel = az_rel_rad * 180/math.pi
    alt_ref = alt_ref_rad  * 180/math.pi

    # subtract az_rel because az_rel direction is opposite the camera reference
    azalt_ref = [known_azalt[0] - az_rel*x_direction, alt_ref]

    return azalt_ref