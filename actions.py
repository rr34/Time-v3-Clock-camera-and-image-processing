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
import camera, awimlib, astropytools


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


def generate_png_with_awim_tag(current_image, rotate_degrees, awim_dictionary):
	# create the info object, add the awim data to the info object, save the png with the info object 
    png_data_container = PIL.PngImagePlugin.PngInfo()

    for key, value in awim_dictionary.items():
        png_data_container.add_text(key, value)
    
    save_filename_string = os.path.splitext(current_image.filename)[0] + ' - awim.png'
    current_image = current_image.rotate(angle=rotate_degrees, expand=True) # rotates CW
    current_image.save(save_filename_string, 'PNG', pnginfo=png_data_container)


def AWIM_generate_tag_from_exif(source_image_path, metadata_source_path, camera_AWIM, AWIMtag_dictionary, \
        elevation_at_Location, tz, known_px, known_px_azart, img_orientation, img_tilt):

    exif_readable = awimlib.get_exif(metadata_source_path)

    if AWIMtag_dictionary['LocationSource'] != 'get from exif GPS' and isinstance(AWIMtag_dictionary['Location'], (list, tuple)):
        pass # allows user to specify location without being overridden by the exif GPS
    elif AWIMtag_dictionary['LocationSource'] == 'get from exif GPS':
        if exif_readable.get('GPSInfo'):
            Location, LocationAltitude = awimlib.get_exif_GPS(exif_readable)
            if Location:
                AWIMtag_dictionary['Location'] = Location
                AWIMtag_dictionary['LocationSource'] = 'DSC exif GPS'
            else:
                AWIMtag_dictionary['LocationSource'] = 'Attempted to get from exif GPS, but was not present or not complete.'
                
            if LocationAltitude:
                AWIMtag_dictionary['LocationAltitude'] = LocationAltitude
                AWIMtag_dictionary['LocationAltitudeSource'] = 'DSC exif GPS'
            else:
                AWIMtag_dictionary['LocationAltitudeSource'] = 'Attempted to get from exif GPS, but was not present or not complete.'
        else:
            AWIMtag_dictionary['LocationSource'] = 'Attempted to get from exif GPS, but GPSInfo was not present at all in exif.'
            AWIMtag_dictionary['LocationAltitudeSource'] = 'Attempted to get from exif GPS, but GPSInfo was not present at all in exif.'
    else:
        print('Error')
    
    if AWIMtag_dictionary['LocationAGLSource'] == 'get from altitude minus terrain elevation':
        LocationAGL = awimlib.get_locationAGL_from_alt_minus_elevation(AWIMtag_dictionary, elevation_at_Location)
        if LocationAGL:
            AWIMtag_dictionary['LocationAGL'] = LocationAGL
            AWIMtag_dictionary['LocationAGLSource'] = 'Subtracted terrain elevation from altitude.'
        else:
            AWIMtag_dictionary['LocationAGLSource'] = 'Attempted to subtract elevation from altitude, but required information was not complete.'

    if AWIMtag_dictionary['CaptureMomentSource'] == 'get from exif':
        UTC_datetime_str, UTC_source = awimlib.UTC_from_exif(exif_readable, tz)
        if UTC_datetime_str:
            AWIMtag_dictionary['CaptureMoment'] = UTC_datetime_str
            AWIMtag_dictionary['CaptureMomentSource'] = UTC_source
        else:
            AWIMtag_dictionary['CaptureMomentSource'] = 'Attempted to get from exif GPS, but was not present or not complete.'

    pixel_map_type, xyangs_model_df, px_model_df = camera_AWIM.generate_xyang_pixel_models\
                                                    (source_image_path, img_orientation, img_tilt)

    AWIMtag_dictionary['PixelMapType'] = pixel_map_type
    AWIMtag_dictionary['AngleModels'] = xyangs_model_df
    AWIMtag_dictionary['PixelModels'] = px_model_df
    
    ref_px, img_px_borders = awimlib.get_ref_px_and_borders(source_image_path, AWIMtag_dictionary['RefPixel'])

    AWIMtag_dictionary['RefPixel'] = ref_px

    if AWIMtag_dictionary['RefPixelAzimuthArtifaeSource'] == 'from known px':
        if isinstance(known_px_azart, str):
            ref_azart_source = 'From celestial object in photo: ' + known_px_azart
            known_px_azart = astropytools.get_AzArt(AWIMtag_dictionary, known_px_azart)

        ref_azart = awimlib.ref_px_from_known_px(AWIMtag_dictionary, known_px, known_px_azart)

    AWIMtag_dictionary['RefPixelAzimuthArtifae'] = ref_azart.tolist()
    AWIMtag_dictionary['RefPixelAzimuthArtifaeSource'] = ref_azart_source

    img_borders_azarts = awimlib.pxs_to_azarts(AWIMtag_dictionary, img_px_borders)
    AWIMtag_dictionary['AzimuthArtifaeBorders'] = img_borders_azarts.tolist()

    img_borders_RADecs = astropytools.AzArts_to_RADecs(AWIMtag_dictionary, img_borders_azarts)
    AWIMtag_dictionary['RADecBorders'] = img_borders_RADecs.tolist()

    ch, cv, ah, av = awimlib.get_pixel_sizes(source_image_path, AWIMtag_dictionary)

    img_xyangs_borders = self.px_xyangs_models_convert(input=np.divide(img_px_borders.reshape(-1,2), img_resize_factor), direction='px_to_xyangs')

    AWIMtag_dictionary_string = awimlib.stringify_tag(AWIMtag_dictionary)
    print('pause here to check')
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
