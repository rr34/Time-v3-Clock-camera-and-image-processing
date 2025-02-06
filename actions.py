from tkinter.filedialog import askopenfilename
import os, shutil
import json
import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
import datetime
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
import camera, awimlib, astropytools, XMPtext, formatters, metadata_tools, DBsqlstatements
# import math
# import PIL
# import pytz


def cam_calibration():
    workingpath = os.path.join(os.getcwd(), 'working')
    image_path = os.path.join(workingpath, 'calimage.jpg')
    cal_file_path = os.path.join(workingpath, 'calspreadsheet.xlsx')

    cam_AWIMtag, filename = camera.generate_camera_AWIM_from_calibration(image_path, cal_file_path)

    file_path = os.path.join(workingpath, filename)
    with open(file_path, 'w') as json_file:
        json.dump(cam_AWIMtag, json_file, indent=4, sort_keys=True)

    return


def generate_metatext_files():
    workingpath = os.path.join(os.getcwd(), 'working')
    for file in os.listdir(workingpath):
        file_path = os.path.join(workingpath, file)
        metadata_dict = metadata_tools.get_metadata(file_path)
        file_base = os.path.splitext(file_path)[0]
        json_file_name = file_base + '.json'
        with open(json_file_name, "w") as text_file:
            json.dump(metadata_dict, text_file, indent=4, sort_keys=True)

    return


def generate_image_tags():
    photoshootID = 'timhouse20220410'
    # 1. Get lists of photo files and entries in the database.
    photoshoot_basenames = DBsqlstatements.get_basenames(photoshootID)
    workingpath = os.path.join(os.getcwd(), 'working')
    imagebases_list = []
    images_list_iterable = []
    for file in os.listdir(workingpath):
        file_type = os.path.splitext(file)[-1]
        file_base = os.path.splitext(file)[0]
        if 'cam_awim.json' in file.lower(): # software could eventually select the correct camera awim from many based on the exif lens information from the images, but for now all photos in the batch need to be taken with the same lens.
            awim_path = os.path.join(workingpath, file)
            with open(awim_path, 'r') as json_file:
                cam_AWIMtag_dictionary = json.load(json_file)
        elif file_type.lower() in ('.jpg', '.png', 'jpeg'):
            file_path = os.path.join(workingpath, file)
            imagebases_list.append(file_base)
            if file_base in photoshoot_basenames:
                images_list_iterable.append((file_path, file_base)) # enough information to match the image file path to the correct unique entry in the database.

    # 2. Match the photo files to entries in the database and identify mismatches.
    matches = list(set(photoshoot_basenames).intersection(imagebases_list))
    lonely_files = list(set(imagebases_list) - set(photoshoot_basenames))
    lonely_shootentries = list(set(photoshoot_basenames) - set(imagebases_list))
    if len(lonely_files) == 0 and len(lonely_shootentries) == 0:
        perfect_match = True
    else:
        perfect_match = False
    if perfect_match:
        print('Photo files and shoot entries match perfectly.')
    else:
        print('Lonely files list: ' + str(lonely_files))
        print('Lonely photoshoot entries list: ' + str(lonely_shootentries))

    # 3. Iterate over the image files using the basename to get the unique corresponding entry in the database.
    for image in images_list_iterable:
        image_path = image[0]
        image_basename = image[1]
        photoshoot_dictionary = DBsqlstatements.get_photo(photoshootID, image_basename)
        if len(photoshoot_dictionary) == 1:
            photoshoot_dictionary = photoshoot_dictionary[0]
        elif len(photoshoot_dictionary) > 1:
            print('Duplicate entry for basename: ' + image_basename)
        elif len(photoshoot_dictionary) < 1:
            print('Some unknown error for : ' + image_basename)
        
        AWIMtag_dict = camera.generate_tag_from_exif_plus_misc(image_path, cam_AWIMtag_dictionary, photoshoot_dictionary)

        # 4. Save each awim tag json file, along with a copy of the image file of the same base name.
        image_filetype = os.path.splitext(image_path)[1]
        new_basename = photoshootID + ' - ' + image_basename
        new_image_path = os.path.join(workingpath, new_basename) + image_filetype
        json_path = os.path.join(workingpath, new_basename) + '.json'

        with open(json_path, "w") as text_file:
            json.dump(AWIMtag_dict, text_file, indent=4, sort_keys=True)

        shutil.copy2(image_path, new_image_path) # todo: generate the tag in place, then rename the file later? copy2 preserves metadata like time stamps

    return


def lightroom_timelapse_XMP_process():
    XMPdirectory = os.path.join(os.getcwd(), 'working')

    # read XMP files
    columns_to_interpolate = ['crs Temperature', 'crs Tint', 'crs Exposure2012', 'crs Contrast2012', 'crs Highlights2012', 'crs Shadows2012', 'crs Whites2012', 'crs Blacks2012', 'crs Texture', 'crs Clarity2012', 'crs Dehaze', 'crs Vibrance', 'crs Saturation']
    XMP_snapshot, lapse_latlng = XMPtext.readXMPfiles(XMPdirectory, columns_to_interpolate)
    XMP2 = XMP_snapshot.copy()
    # set variables for sun and moon calculations
    moments_list = XMP2['exif DateTimeOriginal'].values
    moments_list = formatters.format_datetime(input_datetime=moments_list, direction='from list of ISO 8601 strings')
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
    time_string = formatters.format_datetime(timenow, 'to string for filename')
    filename = f'XMP_step1 {time_string}.csv'
    filepath = os.path.join(XMPdirectory, filename)
    XMP2.to_csv(filepath)

    # write the comma-separated tags to the XMP files
    XMPtext.addTags(XMP_snapshot, XMP2, XMPdirectory)
    print('Completed step 1 labelling XMP files with celestial events.')

    XMP_snapshot, lapse_latlng = XMPtext.readXMPfiles(XMPdirectory, columns_to_interpolate)
    XMP2 = XMP_snapshot.copy() # this seems unnecessary since XMP2 is defined in the next line, but maybe reauired to prevent XMP2 from pointing to XMP_snapshot.
    print('Interpolating the dataframe of XMP values...')
    XMP2 = XMPtext.interpolate(XMP_snapshot, columns_to_interpolate)
    # save dataframe to CSV file
    print('Saving interpolated dataframe to CSV...')
    timenow = datetime.datetime.now()
    time_string = formatters.format_datetime(timenow, 'to string for filename')
    filename = f'XMP_step2 {time_string}.csv'
    filepath = os.path.join(XMPdirectory, filename)
    XMP2.to_csv(filepath)

    # write the new values to the XMP files
    XMPtext.write_values(XMP2, columns_to_interpolate, XMPdirectory)
    print('Completed step 2 interpolating between the keyframes and writing to XMP files.')


# ----- unknown below this line -----
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