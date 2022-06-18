import tkinter
import pickle
import os
import numpy as np
import math
import PIL
from PIL.ExifTags import TAGS, GPSTAGS
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import datetime
import pytz
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
import camera, basic_functions


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


def do_nothing():
    pass


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


def generate_png_with_awim_tag(current_image, rotate_degrees, awim_dictionary):
	# create the info object, add the awim data to the info object, save the png with the info object 
    png_data_container = PIL.PngImagePlugin.PngInfo()

    for key, value in awim_dictionary.items():
        png_data_container.add_text(key, value)
    
    save_filename_string = os.path.splitext(current_image.filename)[0] + ' - awim.png'
    current_image = current_image.rotate(angle=rotate_degrees, expand=True) # rotates CW
    current_image.save(save_filename_string, 'PNG', pnginfo=png_data_container)


def generate_image_with_AWIM_tag(src_img_path, metadata_source_path, tz_default, center_ref, current_camera, img_orientation, img_tilt):

    AWIMtag_dictionary = {'Location': None, 'LocationUnit': None, 'LocationSource': None, \
                            'LocationAltitude': None, 'LocationAltitudeUnit': None, 'LocationAltitudeSource': None, \
                            'LocationAGL': None, 'LocationAGLUnit': None, 'LocationAGLSource': None, \
                            'CaptureMoment': None, 'CaptureMomentUnit': None, 'CaptureMomentSource': None, \
                            'PixelMapType': None, 'CenterPixel': None, 'CenterPixelRef': None, 'CenterAzArt': None, \
                            'PixelModelsFeatures': None, 'AngleModelsFeatures': None, 'PixelBorders': None, 'AngleBorders': None, \
                            'AzimuthArtifaeBorders': None, 'RADecBorders': None, 'RADecUnit': None, \
                            'PixelSizeCenterHorizontal: ': None, 'PixelSizeCenterVertical: ': None, 'PixelSizeUnit': 'Degrees per pixel'}

    exif_present = basic_functions.exif_to_pickle(metadata_source_path)
    if exif_present:
        location, locationAltitude = basic_functions.exif_GPSlatlng_formatted(metadata_source_path)
        UTC_datetime_str, UTC_source = basic_functions.UTC_from_exif(metadata_source_path, tz_default)
    else:
        location = locationAltitude = UTC_datetime_str = False

    if location:
        AWIMtag_dictionary['Location'] = ', '.join(str(i) for i in location)
        AWIMtag_dictionary['LocationUnit'] = 'Latitude, Longitude'
        AWIMtag_dictionary['LocationSource'] = 'DSC exif GPS'
    if locationAltitude:
        AWIMtag_dictionary['LocationAltitude'] = '%f' % locationAltitude
        AWIMtag_dictionary['LocationAltitudeUnit'] = 'Meters above sea level'
        AWIMtag_dictionary['LocationAltitudeSource'] = 'DSC exif GPS'
    if UTC_datetime_str:
        AWIMtag_dictionary['CaptureMoment'] = UTC_datetime_str
        AWIMtag_dictionary['CaptureMomentUnit'] = 'Gregorian New Style Calendar YYYY:MM:DD, Time is UTC HH:MM:SS'
        AWIMtag_dictionary['CaptureMomentSource'] = UTC_source


    # take user input for missing information
    # generate the directional tag information
    # format the directional information and fill in the dictionary

    pixel_map_type, xyangs_model_csv, px_model_csv = current_camera.generate_xyang_pixel_models\
                                                                        (src_img_path, img_orientation, img_tilt)

    img_center_px = basic_functions.do_center_ref(src_img_path, center_ref)

    print('pause here to check')

    px_LT = [0-img_center[0], img_center[1]]
    px_top = [0, img_center[1]]
    px_RT = [img_center[0], img_center[1]]
    px_left = [0-img_center[0], 0]
    px_center = [0, 0]
    px_right = [img_center[0], 0]
    px_LB = [0-img_center[0], 0-img_center[1]]
    px_bottom = [0, 0-img_center[1]]
    px_RB = [img_center[0], 0-img_center[1]]
    img_px_borders = np.concatenate((px_LT, px_top, px_RT, px_left, px_center, px_right, px_LB, px_bottom, px_RB)).reshape(-1,2)
    img_xyangs_borders = self.px_xyangs_models_convert(input=np.divide(img_px_borders, img_resize_factor), direction='px_to_xyangs')

    pxs_LRUD = np.array([[-1,0],[1,0],[0,-1],[0,1]])
    img_xyangs_LRUD = self.px_xyangs_models_convert(input=np.divide(pxs_LRUD, img_resize_factor), direction='px_to_xyangs')
    x_pxsize_degperhundredpx = 100 * abs(img_xyangs_LRUD[0,0]-img_xyangs_LRUD[1,0]) / abs(pxs_LRUD[0,0]-pxs_LRUD[1,0])
    y_pxsize_degperhundredpx = 100 * abs(img_xyangs_LRUD[2,1]-img_xyangs_LRUD[3,1]) / abs(pxs_LRUD[2,1]-pxs_LRUD[3,1])
    px_size_center = [x_pxsize_degperhundredpx, y_pxsize_degperhundredpx]

    center_ref_string = ', '.join(str(i) for i in center_ref)
    azalt_ref_string = ', '.join(str(i) for i in azalt_ref)
    px_borders_string = ', '.join(str(i) for i in img_px_borders)
    xyangs_borders_string = ', '.join(str(i) for i in img_xyangs_borders)
    px_size_center_str = ', '.join(str(i) for i in px_size_center)
