import pickle
import os, io, ast, re
import math
import numpy as np
import PIL
from PIL.ExifTags import TAGS, GPSTAGS
import datetime
import pytz
import pandas as pd
import astropytools


def generate_empty_AWIMtag_dictionary(default_units=True):
    AWIMtag_dictionary = {}
    AWIMtag_dictionary['Location'] = None
    AWIMtag_dictionary['LocationUnit'] = 'Latitude, Longitude to 6 decimal places, ~11cm'
    AWIMtag_dictionary['LocationSource'] = None
    AWIMtag_dictionary['LocationAltitude'] = None
    AWIMtag_dictionary['LocationAltitudeUnit'] = 'Meters above sea level'
    AWIMtag_dictionary['LocationAltitudeSource'] = None
    AWIMtag_dictionary['LocationAGL'] = None
    AWIMtag_dictionary['LocationAGLUnit'] = 'Meters above ground level'
    AWIMtag_dictionary['LocationAGLSource'] = None
    AWIMtag_dictionary['CaptureMoment'] = None
    AWIMtag_dictionary['CaptureMomentUnit'] = 'Gregorian New Style Calendar YYYY:MM:DD, Time is UTC HH:MM:SS'
    AWIMtag_dictionary['CaptureMomentSource'] = None
    AWIMtag_dictionary['PixelAngleModelsType'] = None
    AWIMtag_dictionary['RefPixel'] = None
    AWIMtag_dictionary['RefPixelCoordType'] = 'top-left is (0,0) so standard.'
    AWIMtag_dictionary['RefPixelAzimuthArtifae'] = None
    AWIMtag_dictionary['RefPixelAzimuthArtifaeSource'] = None
    AWIMtag_dictionary['RefPixelAzimuthArtifaeUnit'] = 'Degrees'
    AWIMtag_dictionary['AnglesModel'] = None
    AWIMtag_dictionary['PixelsModel'] = None
    AWIMtag_dictionary['BorderPixels'] = None
    AWIMtag_dictionary['BorderAngles'] = None
    AWIMtag_dictionary['BordersAzimuthArtifae'] = None
    AWIMtag_dictionary['BordersRADec'] = None
    AWIMtag_dictionary['RADecUnit'] = 'ICRS J2000 Epoch'
    AWIMtag_dictionary['PixelSizeCenterHorizontalVertical'] = None
    AWIMtag_dictionary['PixelSizeAverageHorizontalVertical'] = None
    AWIMtag_dictionary['PixelSizeUnit'] = 'Pixels per Degree'

    if not default_units:
        for key in AWIMtag_dictionary:
            AWIMtag_dictionary[key] = None

    return AWIMtag_dictionary


def format_datetime(input_datetime_UTC, direction):
    datetime_format = "%Y:%m:%d %H:%M:%S" # directly from exif documentation
    
    if direction == 'to string':
        if isinstance(input_datetime_UTC, datetime.datetime):
            output = input_datetime_UTC.strftime(datetime_format)
        elif isinstance(input_datetime_UTC, np.datetime64):
            pass # TODO convert the format to datetime_format, necessary?

    elif direction == 'from string':
        datetime_object = datetime.datetime.strptime(input_datetime_UTC, datetime_format)
        numpy_datetime_format = "%Y-%m-%dT%H:%M:%S" # from numpy documentation
        datetime_string_for_numpy = datetime.datetime.strftime(datetime_object, numpy_datetime_format)
        output = np.datetime64(datetime_string_for_numpy)

    return output


def get_exif(metadata_source_path, save_exif_text_file=False):
    metadata_src_type = os.path.splitext(metadata_source_path)[-1]
    current_image = PIL.Image.open(metadata_source_path)
    img_exif = current_image._getexif()

    if img_exif:
        img_exif_readable = {}
        for key, value in img_exif.items():
            decode = TAGS.get(key,key)
            if key != 34853:
                img_exif_readable[decode] = value
            else:
                GPS_dict_readable = {}
                for GPSkey, GPSvalue in value.items():
                    GPSdecode = GPSTAGS.get(GPSkey,GPSkey)
                    GPS_dict_readable[GPSdecode] = GPSvalue
                img_exif_readable[decode] = GPS_dict_readable

        # with open((os.path.splitext(metadata_source_path)[0] + ' - exif' + '.pickle'), 'wb') as exif_pickle:
        #     pickle.dump(img_exif, exif_pickle, 5)

        if save_exif_text_file:
            img_exif_readable_str = format_dictionary(img_exif_readable, 'string')
            img_exif_readable_str = img_exif_readable_str.replace("',", "',\n")
            savename = os.path.splitext(metadata_source_path)[0]
            with open(savename + ' - exif_readable.txt', 'w') as f:
                f.write(img_exif_readable_str)

        return img_exif, img_exif_readable
    else:
        return False


def extract_latlng_from_exif_GPS(exif_readable):
    lat_sign = lng_sign = GPS_latlng = GPS_alt = False

    if exif_readable['GPSInfo'].get('GPSLatitudeRef') == 'N':
        lat_sign = 1
    elif exif_readable['GPSInfo'].get('GPSLatitudeRef') == 'S':
        lat_sign = -1
    if exif_readable['GPSInfo'].get('GPSLongitudeRef') == 'E':
        lng_sign = 1
    elif exif_readable['GPSInfo'].get('GPSLongitudeRef') == 'W':
        lng_sign = -1
    if lat_sign and lng_sign and exif_readable['GPSInfo'].get('GPSLatitude') and exif_readable['GPSInfo'].get('GPSLongitude'):
        img_lat = lat_sign * float(exif_readable['GPSInfo']['GPSLatitude'][0] + exif_readable['GPSInfo']['GPSLatitude'][1]/60 + exif_readable['GPSInfo']['GPSLatitude'][2]/3600)
        img_lng = lng_sign * float(exif_readable['GPSInfo']['GPSLongitude'][0] + exif_readable['GPSInfo']['GPSLongitude'][1]/60 + exif_readable['GPSInfo']['GPSLongitude'][2]/3600)
        GPS_latlng = [img_lat, img_lng]

    if exif_readable['GPSInfo'].get('GPSAltitudeRef') and exif_readable['GPSInfo'].get('GPSAltitude'):
        GPSAltitudeRef = exif_readable['GPSInfo']['GPSAltitudeRef'].decode()
        if GPSAltitudeRef == '\x00':
            GPS_alt_sign = 1
        else:
            print('Elevation is something unusual, probably less than zero like in Death Valley or something. Look here.')
            GPS_alt_sign = -1
        GPS_alt = GPS_alt_sign * float(exif_readable['GPSInfo']['GPSAltitude'])

    return GPS_latlng, GPS_alt

    
def UTC_from_exif(exif_readable, tz_default):
    exif_UTC = False
    UTC_source = False
    UTC_datetime_str = False
    exif_datetime_format = "%Y:%m:%d %H:%M:%S"

    # 1. use the GPS time tag if present and ignore the leap seconds
    if exif_readable.get('GPSInfo') and exif_readable['GPSInfo'].get('GPSDateStamp') and exif_readable['GPSInfo'].get('GPSTimeStamp'):
        GPS_datestamp = exif_readable['GPSInfo']['GPSDateStamp']
        GPS_timestamp = exif_readable['GPSInfo']['GPSTimeStamp']
        GPSyear = int(GPS_datestamp.split(':')[0])
        GPSmonth = int(GPS_datestamp.split(':')[1])
        GPSday = int(GPS_datestamp.split(':')[2])
        GPShour = int(GPS_timestamp[0])
        GPSminute = int(GPS_timestamp[1])
        GPSsecond = int(GPS_timestamp[2])
        exif_UTC = datetime.datetime(year = GPSyear, month = GPSmonth, day = GPSday, hour = GPShour, minute = GPSminute, second = GPSsecond)
        UTC_source = 'exif GPSDateStamp and exif GPSTimeStamp'
    
    # 2. use the datetimeoriginal with either UTCOffset tag or default timezone
    elif exif_readable.get('DateTimeOriginal'):
        exif_datetime = exif_readable['DateTimeOriginal']
        exif_datetime_object_naive = datetime.datetime.strptime(exif_datetime, exif_datetime_format)

        if  exif_readable.get('OffsetTimeOriginal'):
            exif_UTC_offset_minutes = 60 * int(exif_readable['OffsetTimeOriginal'].split(':')[0]) + int(exif_readable['OffsetTimeOriginal'].split(':')[1])
            UTC_offset_timedelta = datetime.timedelta(minutes = exif_UTC_offset_minutes)
            exif_UTC = exif_datetime_object_naive - UTC_offset_timedelta
            UTC_source = 'exif DateTimeOriginal and exif OffsetTimeOriginal'
        else:
            exif_datetime_object_localized = tz_default.localize(exif_datetime_object_naive)
            exif_UTC = exif_datetime_object_localized.astimezone(pytz.utc)
            UTC_source = 'exif DateTimeOriginal adjusted with user-entered timezone'

    if exif_UTC:
        UTC_datetime_str = format_datetime(exif_UTC, 'to string')

    return UTC_datetime_str, UTC_source


def get_ref_px_and_borders(source_image_path, ref_px):
    source_image = PIL.Image.open(source_image_path)
    img_dimensions = source_image.size
    max_img_index = np.subtract(img_dimensions, 1)

    if ref_px == 'center, get from image':
        img_center = np.divide(max_img_index, 2).tolist()
        ref_px = img_center

        left = -img_center[0]
        right = img_center[0]
        top = img_center[1]
        bottom = -img_center[1]
    
    img_borders_pxs = np.array([[left,top,0,top,right,top], [left,0,0,0,right,0], [left,bottom,0,bottom,right,bottom]])

    return ref_px, img_borders_pxs


def get_locationAGL_from_alt_minus_elevation(AWIMtag_dictionary, elevation_at_Location):
    if 0:
        pass # subtract the elevation_at_Location from LocationAltitude
    else:
        LocationAGL = False
    
    return LocationAGL


def format_dictionary(any_dictionary, output_type):
    any_dictionary_ofstrings = {}
    any_dictionary_asstring = ''

    for key, value in any_dictionary.items():
        print(type(value))
        if isinstance(value, (list, tuple)):
            value_string = ', '.join(str(i) for i in value)
        elif isinstance(value, (int, float)) or (value is None):
            value_string = str(value)
        elif isinstance(value, pd.DataFrame):
            value_string = value.to_csv(index_label='features')
        elif isinstance(value, dict):
            value_string = format_dictionary(value, output_type)
        elif isinstance(value, str):
            value_string = value
        elif isinstance(value, bytes):
            try:
                value_string = str(value.decode())
            except (UnicodeDecodeError, AttributeError):
                pass
        elif isinstance(value, PIL.TiffImagePlugin.IFDRational):
            if value.denominator != 0:
                value_string = str(value.numerator / value.denominator)
            else:
                value_string = str(float('nan'))
        else:
            value_string = 'unexpected data type'

        if output_type == 'dictionary as text':
            any_dictionary_ofstrings[key] = value_string
        elif output_type == 'string':
            value_string = value_string.replace("\\", "") # TODO doesn't work for some reason
            value_string = key + ': ' + value_string + '\n'
            any_dictionary_asstring += value_string

    if output_type == 'dictionary as text':
        return str(any_dictionary_ofstrings)
    elif output_type == 'string':
        return any_dictionary_asstring


def de_stringify_tag(AWIMtag_dictionary_string):
    AWIMtag_dictionary_string = AWIMtag_dictionary_string.replace("AWIMstart", "")
    AWIMtag_dictionary_string = AWIMtag_dictionary_string.replace("AWIMend", "")
    AWIMtag_dictionary_string = AWIMtag_dictionary_string.replace("',\n", "',")
    AWIMtag_dictionary_ofstrings = ast.literal_eval(AWIMtag_dictionary_string)
    AWIMtag_dictionary = {}
    for key, value in AWIMtag_dictionary_ofstrings.items():
        if value is None:
            AWIMtag_dictionary[key] = None
        elif (key == 'PixelsModel') or (key == 'AnglesModel'): # these two started as dataframes
            AWIMtag_dictionary[key] = pd.read_csv(io.StringIO(value), index_col=0)
        elif key == 'CaptureMoment':
            AWIMtag_dictionary[key] = value
        elif (',' in value) and not (re.search('[a-zA-Z]', value)): # evaluate as a list
            AWIMtag_dictionary[key] = [float(list_value) for list_value in value.split(',')]
        elif (',' not in value) and ('.' not in value) and not (re.search('[a-zA-Z]', value)): # evaluate as an int
            AWIMtag_dictionary[key] = int(value)
        elif (',' not in value) and ('.' in value) and not (re.search('[a-zA-Z]', value)): # evaluate as a float
            AWIMtag_dictionary[key] = float(value)
        else:
            AWIMtag_dictionary[key] = value

    return AWIMtag_dictionary


def pxs_to_xyangs(AWIMtag_dictionary, pxs):
    if isinstance(pxs, (list, tuple)): # models require numpy arrays
        pxs = np.asarray(pxs)

    input_shape = pxs.shape
    angs_direction = np.where(pxs < 0, -1, 1) # models are positive values only. Save sign. Same sign for xyangs

    pxs = np.abs(pxs).reshape(-1,2)

    if AWIMtag_dictionary['PixelAngleModelsType'] == '3d_degree_poly_fit_abs_from_center':
        pxs_poly = np.zeros((pxs.shape[0], 9))
        pxs_poly[:,0] = pxs[:,0]
        pxs_poly[:,1] = pxs[:,1]
        pxs_poly[:,2] = np.square(pxs[:,0])
        pxs_poly[:,3] = np.multiply(pxs[:,0], pxs[:,1])
        pxs_poly[:,4] = np.square(pxs[:,1])
        pxs_poly[:,5] = np.power(pxs[:,0], 3)
        pxs_poly[:,6] = np.multiply(np.square(pxs[:,0]), pxs[:,1])
        pxs_poly[:,7] = np.multiply(pxs[:,0], np.square(pxs[:,1]))
        pxs_poly[:,8] = np.power(pxs[:,1], 3)

    xang_predict_coeff = AWIMtag_dictionary['AnglesModel'].loc[['xang_predict']].values[0]
    yang_predict_coeff = AWIMtag_dictionary['AnglesModel'].loc[['yang_predict']].values[0]

    xyangs = np.zeros(pxs.shape)
    xyangs[:,0] = np.dot(pxs_poly, xang_predict_coeff)
    xyangs[:,1] = np.dot(pxs_poly, yang_predict_coeff)

    xyangs_pretty = np.multiply(xyangs.reshape(input_shape), angs_direction)

    return xyangs_pretty


def xyangs_to_azarts(AWIMtag_dictionary, xyangs, ref_azart_override=False):

    # prepare to convert xyangs to azarts. Already have angs_direction from above. abs of xangs only, keep negative yangs
    input_shape = xyangs.shape
    angs_direction = np.where(xyangs < 0, -1, 1)
    xyangs = xyangs.reshape(-1,2)
    angs_direction = angs_direction.reshape(-1,2)
    xyangs[:,0] = np.abs(xyangs[:,0])
    xyangs *= math.pi/180
    if isinstance(ref_azart_override, (list, tuple, np.ndarray)):
        ref_azart_rad = np.multiply(ref_azart_override, math.pi/180)
    else:
        ref_azart_rad = np.multiply(AWIMtag_dictionary['RefPixelAzimuthArtifae'], math.pi/180)

    # see photoshop diagram of sphere, circles, and triangles for variable names
    xang_compliment = np.subtract(math.pi/2, xyangs[:,0]) # always (+) because xang < 90
    d1 = 1*np.cos(xang_compliment) # always (+)
    r2 = 1*np.sin(xang_compliment) # always (+)
    ang_totalsmallcircle = np.add(ref_azart_rad[1], xyangs[:,1]) # -180 to 180
    d2_ = np.multiply(np.cos(ang_totalsmallcircle), r2) # (-) for ang_totalsmallcircle > 90 or < -90, meaning px behind observer
    art_seg_ = np.multiply(np.sin(ang_totalsmallcircle), r2) # (-) for (-) ang_totalsmallcircle
    arts = np.arcsin(art_seg_) # (-) for (-) art_seg_
    az_rel = np.subtract(math.pi/2, np.arctan(np.divide(d2_, d1))) # d2 (-) for px behind observer and therefore az_rel > 90 because will subtract (-) atan
    az_rel = np.multiply(az_rel, angs_direction[:,0])
    azs = np.mod(np.add(ref_azart_rad[0], az_rel), 2*math.pi)

    azarts = np.zeros(xyangs.shape)
    azarts[:,0] = np.multiply(azs, 180/math.pi)
    azarts[:,1] = np.multiply(arts, 180/math.pi)

    azarts = azarts.reshape(input_shape)

    return azarts


# TODO this function is untested 1 jul 2022
def azarts_to_xyangs(AWIMtag_dictionary, azarts):
    if isinstance(azarts, list):
        azarts = np.asarray(azarts)

    input_shape = azarts.shape
    azarts = azarts.reshape(-1,2)

    ref_px_azart = AWIMtag_dictionary['RefPixelAzimuthArtifae']

    # find az_rels, convert to -180 < x <= 180, then abs value + direction matrix
    # also need az_rel compliment angle + store which are behind camera
    # then to radians
    simple_subtract = np.subtract(azarts[:,0], ref_px_azart[0])
    big_angle_correction = np.where(simple_subtract > 0, -360, 360)
    az_rel = np.where(np.abs(simple_subtract) <= 180, simple_subtract, np.add(simple_subtract, big_angle_correction))
    az_rel_direction = np.where(az_rel < 0, -1, 1)
    az_rel_abs = np.abs(az_rel)
    az_rel_behind_observer = np.where(az_rel_abs <= 90, False, True) # true if point is behind observer - assume because camera pointed up very high past zenith or pointed very low below nether-zenith
    # Note: cannot allow az_rel_compliment (and therefore d2) to be negative because must simple_ang_totalsmallcircle be (-) if art_seg_ is (-), which is good.
    az_rel_compliment = np.where(az_rel_abs <= 90, np.subtract(90, az_rel_abs), np.subtract(az_rel_abs, 90)) # 0 to 90 angle from line perpendicular to az
    az_rel_compliment_rad = np.multiply(az_rel_compliment, math.pi/180) # (+) only, 0 to 90
    ref_px_azart_rad = np.multiply(ref_px_azart, math.pi/180)

    # artifae direction matrix, then artifaes to radians, keep sign just convert to radian
    art_direction = np.where(azarts[:,1] < 0, -1, 1)
    art_rad = np.multiply(azarts[:,1], math.pi/180)

    # trigonometry, see photoshop diagrams for variable descriptions. segment ending with underscore_ means "can be negative distance"
    art_seg_ = np.sin(art_rad) # notice: (-) for (-) arts
    d3 = np.cos(art_rad) # (+) only, because arts are always -90 to 90
    d2 = np.multiply(np.sin(az_rel_compliment_rad), d3) # (+) only
    d1 = np.multiply(np.cos(az_rel_compliment_rad), d3) # (+) only because az_rel_compliment_rad is 0 to 90
    r2 = np.sqrt(np.square(d2), np.square(art_seg_))
    xang_abs = np.subtract(math.pi/2, np.arccos(d1)) # TODO? what if xang is actually > 90? would be unusual, difficult to combine with large yang
    xang = np.multiply(xang_abs, az_rel_direction)
    pt1_art_ = np.multiply(r2, np.sin(ref_px_azart_rad[1])) # (-) for (-) cam_arts, which is good
    lower_half = np.where(art_seg_ < pt1_art_, True, False) # true if px is below middle of photo
    ang_smallcircle_fromhorizon = np.arctan(np.divide(art_seg_, d2)) # -90 to 90, (-) for (-) art_seg_, bc d2 always (+)
    # for yang, if in front, simple, but behind observer, the angle from must be subtracted from 180 or -180 because different angle meaning see photoshop diagram
    ang_totalsmallcircle = np.where(np.logical_not(az_rel_behind_observer), ang_smallcircle_fromhorizon, np.subtract(np.multiply(art_direction, math.pi), ang_smallcircle_fromhorizon))
    yang = np.subtract(ang_totalsmallcircle, ref_px_azart_rad[1]) # simply subtract because |ang_totalsmallcircle| < 180 AND |center_azart[1]| < 90 AND if |ang_totalsmallcircle| > 90, then they are same sign

    xy_angs = np.zeros(input_shape)
    xy_angs[:,0] = np.multiply(xang, 180/math.pi)
    xy_angs[:,1] = np.multiply(yang, 180/math.pi)

    return xy_angs


# TODO this function is untested 1 jul 2022
def xyangs_to_pxs(AWIMtag_dictionary, xy_angs):

    input_shape = xy_angs.shape

    xy_angs_direction = np.where(xy_angs < 0, -1, 1)
    xy_angs_abs = np.abs(xy_angs)

    if AWIMtag_dictionary['PixelAngleModelsType'] == '3d_degree_poly_fit_abs_from_center':
        xy_angs_poly = np.zeros((xy_angs.shape[0], 9))
        xy_angs_poly[:,0] = xy_angs_abs[:,0]
        xy_angs_poly[:,1] = xy_angs_abs[:,1]
        xy_angs_poly[:,2] = np.square(xy_angs_abs[:,0])
        xy_angs_poly[:,3] = np.multiply(xy_angs_abs[:,0], xy_angs_abs[:,1])
        xy_angs_poly[:,4] = np.square(xy_angs_abs[:,1])
        xy_angs_poly[:,5] = np.power(xy_angs_abs[:,0], 3)
        xy_angs_poly[:,6] = np.multiply(np.square(xy_angs_abs[:,0]), xy_angs_abs[:,1])
        xy_angs_poly[:,7] = np.multiply(xy_angs_abs[:,0], np.square(xy_angs_abs[:,1]))
        xy_angs_poly[:,8] = np.power(xy_angs_abs[:,1], 3)

    pxs = np.zeros(input_shape)
    px_models_df = AWIMtag_dictionary['PixelsModel']
    x_px_predict_coeff = px_models_df['']
    pxs[:,0] = np.dot(xy_angs_poly, self.x_px_predict_coeff)
    pxs[:,1] = np.dot(xy_angs_poly, self.y_px_predict_coeff)

    pxs = np.multiply(pxs, xy_angs_direction)

    return pxs


# TODO this function is untested 1 jul 2022
def px_coord_convert(input_pxs, input_type, output_type):
    if ('top-left' in input_type) and ('center' in output_type):
        pxs[:,0] = pxs[:,0] + self.center_KVpx[0]
        pxs[:,1] = pxs[:,1] + self.center_KVpx[1]


def ref_px_from_known_px(AWIMtag_dictionary, known_px, known_px_azart):
    xy_ang = pxs_to_xyangs(AWIMtag_dictionary, known_px)

    xy_ang *= -1

    ref_px_azart = xyangs_to_azarts(AWIMtag_dictionary, xy_ang, ref_azart_override=known_px_azart)

    return ref_px_azart


def pxs_to_azarts(AWIMtag_dictionary, pxs):
    xy_angs = pxs_to_xyangs(AWIMtag_dictionary, pxs)

    azarts = xyangs_to_azarts(AWIMtag_dictionary, xy_angs)

    return azarts


def get_pixel_sizes(source_image_path, AWIMtag_dictionary):
    small_px = 10
    little_cross_LRUD = np.array([-small_px,0,small_px,0,0,small_px,0,-small_px]).reshape(-1,2)
    little_cross_angs = pxs_to_xyangs(AWIMtag_dictionary, little_cross_LRUD)
    px_size_center_horizontal = (abs(little_cross_LRUD[0,0]) + abs(little_cross_LRUD[1,0])) / (abs(little_cross_angs[0,0]) + abs(little_cross_angs[1,0]))
    px_size_center_vertical = (abs(little_cross_LRUD[2,1]) + abs(little_cross_LRUD[3,1])) / (abs(little_cross_angs[2,1]) + abs(little_cross_angs[3,1]))

    source_image = PIL.Image.open(source_image_path)
    img_dimensions = source_image.size

    dimensions = source_image.size
    border_angles = np.array(AWIMtag_dictionary['BorderAngles'])
    horizontal_angle_width = abs(border_angles[1,0]) + abs(border_angles[1,4])
    vertical_angle_width = abs(border_angles[0,3]) + abs(border_angles[2,3])
    px_size_average_horizontal = dimensions[0] / horizontal_angle_width
    px_size_average_vertical = dimensions[1] / vertical_angle_width

    return [px_size_center_horizontal, px_size_center_vertical], [px_size_average_horizontal, px_size_average_vertical]


def generate_tag_from_exif_plus_misc(source_image_path, metadata_source_path, camera_AWIM, AWIMtag_dictionary, \
        elevation_at_location, tz, known_px, known_px_azart, img_orientation, img_tilt):

    lat_lng_digits = 6
    GPS_alt_digits = 1
    AGL_digits = 2
    pixel_digits = 1
    degrees_digits = 2
    hourangle_digits = 3
    
    exif_raw, exif_readable = get_exif(metadata_source_path)

    if AWIMtag_dictionary['LocationSource'] != 'get from exif GPS' and isinstance(AWIMtag_dictionary['Location'], (list, tuple)):
        pass # allows user to specify location without being overridden by the exif GPS
    elif AWIMtag_dictionary['LocationSource'] == 'get from exif GPS':
        if exif_readable.get('GPSInfo'):
            location, location_altitude = extract_latlng_from_exif_GPS(exif_readable)
            if location:
                AWIMtag_dictionary['Location'] = [round(f, lat_lng_digits) for f in location]
                AWIMtag_dictionary['LocationSource'] = 'DSC exif GPS'
            else:
                AWIMtag_dictionary['LocationSource'] = 'Attempted to get from exif GPS, but was not present or not complete.'
                
            if location_altitude:
                AWIMtag_dictionary['LocationAltitude'] = round(location_altitude, GPS_alt_digits)
                AWIMtag_dictionary['LocationAltitudeSource'] = 'DSC exif GPS'
            else:
                AWIMtag_dictionary['LocationAltitudeSource'] = 'Attempted to get from exif GPS, but was not present or not complete.'
        else:
            AWIMtag_dictionary['LocationSource'] = 'Attempted to get from exif GPS, but GPSInfo was not present at all in exif.'
            AWIMtag_dictionary['LocationAltitudeSource'] = 'Attempted to get from exif GPS, but GPSInfo was not present at all in exif.'
    else:
        print('Error')
    
    if AWIMtag_dictionary['LocationAGLSource'] == 'get from altitude minus terrain elevation':
        location_AGL = get_locationAGL_from_alt_minus_elevation(AWIMtag_dictionary, elevation_at_location)
        if location_AGL:
            AWIMtag_dictionary['LocationAGL'] = [round(f, AGL_digits) for f in location_AGL]
            AWIMtag_dictionary['LocationAGLSource'] = 'Subtracted terrain elevation from altitude.'
        else:
            AWIMtag_dictionary['LocationAGLSource'] = 'Attempted to subtract elevation from altitude, but required information was not complete.'

    if AWIMtag_dictionary['CaptureMomentSource'] == 'get from exif':
        UTC_datetime_str, UTC_source = UTC_from_exif(exif_readable, tz)
        if UTC_datetime_str:
            AWIMtag_dictionary['CaptureMoment'] = UTC_datetime_str
            AWIMtag_dictionary['CaptureMomentSource'] = UTC_source
        else:
            AWIMtag_dictionary['CaptureMomentSource'] = 'Attempted to get from exif GPS, but was not present or not complete.'

    pixel_map_type, xyangs_model_df, px_model_df = camera_AWIM.generate_xyang_pixel_models\
                                                    (source_image_path, img_orientation, img_tilt)
    AWIMtag_dictionary['PixelAngleModelsType'] = pixel_map_type
    AWIMtag_dictionary['AnglesModel'] = xyangs_model_df
    AWIMtag_dictionary['PixelsModel'] = px_model_df
    
    ref_px, img_borders_pxs = get_ref_px_and_borders(source_image_path, AWIMtag_dictionary['RefPixel'])
    AWIMtag_dictionary['RefPixel'] = [round(f, pixel_digits) for f in ref_px]
    AWIMtag_dictionary['BorderPixels'] = img_borders_pxs.round(pixel_digits).tolist()

    if AWIMtag_dictionary['RefPixelAzimuthArtifaeSource'] == 'from known px':
        if isinstance(known_px_azart, str):
            ref_azart_source = 'From celestial object in photo: ' + known_px_azart
            known_px_azart = astropytools.get_AzArt(AWIMtag_dictionary, known_px_azart)

        ref_azart = ref_px_from_known_px(AWIMtag_dictionary, known_px, known_px_azart)
    AWIMtag_dictionary['RefPixelAzimuthArtifae'] = ref_azart.round(degrees_digits).tolist()
    AWIMtag_dictionary['RefPixelAzimuthArtifaeSource'] = ref_azart_source

    img_borders_angs = pxs_to_xyangs(AWIMtag_dictionary, img_borders_pxs)
    AWIMtag_dictionary['BorderAngles'] = img_borders_angs.round(degrees_digits).tolist()

    img_borders_azarts = pxs_to_azarts(AWIMtag_dictionary, img_borders_pxs)
    AWIMtag_dictionary['BordersAzimuthArtifae'] = img_borders_azarts.round(degrees_digits).tolist()

    img_borders_RADecs = astropytools.AzArts_to_RADecs(AWIMtag_dictionary, img_borders_azarts)
    img_borders_RADecs[:,[0,2,4]] = img_borders_RADecs[:,[0,2,4]].round(hourangle_digits)
    img_borders_RADecs[:,[1,3,5]] = img_borders_RADecs[:,[1,3,5]].round(degrees_digits)
    AWIMtag_dictionary['BordersRADec'] = img_borders_RADecs.tolist()

    px_size_center, px_size_average = get_pixel_sizes(source_image_path, AWIMtag_dictionary)
    AWIMtag_dictionary['PixelSizeCenterHorizontalVertical'] = [round(f, pixel_digits) for f in px_size_center]
    AWIMtag_dictionary['PixelSizeAverageHorizontalVertical'] = [round(f, pixel_digits) for f in px_size_average]

    AWIMtag_dictionary_string = format_dictionary(AWIMtag_dictionary, 'dictionary as text')

    return AWIMtag_dictionary, AWIMtag_dictionary_string


# put the AWIMtag in the comment field of the image exif and re-attach the exif to the image
def add_AWIMtag_to_exif():
	if img_exif_raw.get(37510):
		user_comments = img_exif_raw[37510]
	else:
		user_comments = ''
	img_exif_raw[37510] = user_comments + 'AWIMstart' + cam_AWIMtag_string + 'AWIMend'