import pickle
import os, io, ast, re
import numpy as np
import PIL
from PIL.ExifTags import TAGS, GPSTAGS
import datetime
import pytz
import pandas as pd


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


def exif_to_pickle(image_path):
    current_image = PIL.Image.open(image_path)
    img_exif = current_image._getexif()

    if img_exif:
        img_exif_readable = {}
        for key in img_exif.keys():
            decode = TAGS.get(key,key)
            if key != 34853:
                img_exif_readable[decode] = img_exif[key]
            else:
                GPS_dict_readable = {}
                for GPSkey in img_exif[34853].keys():
                    GPSdecode = GPSTAGS.get(GPSkey,GPSkey)
                    GPS_dict_readable[GPSdecode] = img_exif[34853][GPSkey]
                img_exif_readable[decode] = GPS_dict_readable

        exifs = [img_exif, img_exif_readable]

        with open((os.path.splitext(image_path)[0] + '.exifpickle'), 'wb') as image_exif_pickle:
            pickle.dump(exifs, image_exif_pickle, 5)

        return True
    else:
        return False


def exif_GPSlatlng_formatted(image_path):
    GPS_latlng = False
    GPS_alt = False
    with open((os.path.splitext(image_path)[0] + '.exifpickle'), 'rb') as image_exif_pickle:
        exif_readable = pickle.load(image_exif_pickle)[1]

    if exif_readable.get('GPSInfo'):
        GPS_location_present = True
        if exif_readable['GPSInfo']['GPSLatitudeRef'] == 'N':
            lat_sign = 1
        elif exif_readable['GPSInfo']['GPSLatitudeRef'] == 'S':
            lat_sign = -1
        else:
            GPS_location_present = False

        if exif_readable['GPSInfo']['GPSLongitudeRef'] == 'E':
            lng_sign = 1
        elif exif_readable['GPSInfo']['GPSLongitudeRef'] == 'W':
            lng_sign = -1
        else:
            GPS_location_present = False

        if GPS_location_present:
            img_lat = lat_sign*float(exif_readable['GPSInfo']['GPSLatitude'][0] + exif_readable['GPSInfo']['GPSLatitude'][1]/60 + exif_readable['GPSInfo']['GPSLatitude'][2]/3600)
            img_lng = lng_sign*float(exif_readable['GPSInfo']['GPSLongitude'][0] + exif_readable['GPSInfo']['GPSLongitude'][1]/60 + exif_readable['GPSInfo']['GPSLongitude'][2]/3600)
            GPS_latlng = (img_lat, img_lng)

        GPSAltitudeRef = exif_readable['GPSInfo']['GPSAltitudeRef'].decode()
        if GPSAltitudeRef == '\x00':
            GPS_alt_sign = 1
        else:
            print('Elevation is something unusual, probably less than zero like in Death Valley or something. Look here.')
            GPS_alt_sign = -1
        GPS_alt = GPS_alt_sign * float(exif_readable['GPSInfo']['GPSAltitude'])

    return GPS_latlng, GPS_alt

    
def UTC_from_exif(image_path, tz_default):
    exif_UTC = False
    UTC_source = False
    UTC_datetime_str = False
    exif_datetime_format = "%Y:%m:%d %H:%M:%S"
    with open((os.path.splitext(image_path)[0] + '.exifpickle'), 'rb') as image_exif_pickle:
        exif_readable = pickle.load(image_exif_pickle)[1]

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


def do_center_px(image_source_path, center_px):
    source_image = PIL.Image.open(image_source_path)

    img_dimensions = source_image.size

    max_img_index = np.subtract(img_dimensions, 1)
    img_center = np.divide(max_img_index, 2).tolist()
    if center_px == 'center':
        center_px = img_center

    return center_px

def get_locationAGL():
    if 0:
        pass # attempt to get user input here, also unit and specify source
    else:
        LocationAGL = False
    
    return LocationAGL

def stringify_tag(AWIMtag_dictionary):
    AWIMtag_dictionary_ofstrings = {}
    for key, value in AWIMtag_dictionary.items():
        if isinstance(value, (list, tuple)):
            AWIMtag_dictionary_ofstrings[key] = ', '.join(str(i) for i in value)
        elif isinstance(value, (int, float)):
            AWIMtag_dictionary_ofstrings[key] = str(value)
        elif isinstance(value, pd.DataFrame):
            AWIMtag_dictionary_ofstrings[key] = value.to_csv(index_label='features')
        else:
            AWIMtag_dictionary_ofstrings[key] = value

    AWIMtag_dictionary_string = str(AWIMtag_dictionary_ofstrings)

    return AWIMtag_dictionary_string


def de_stringify_tag(AWIMtag_dictionary_string):
    AWIMtag_dictionary_ofstrings = ast.literal_eval(AWIMtag_dictionary_string)
    AWIMtag_dictionary = {}
    for key, value in AWIMtag_dictionary_ofstrings.items():
        if value is None:
            AWIMtag_dictionary[key] = None
        elif (key == 'PixelModels') or (key == 'AngleModels'):
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


# BELOW HERE IS UN-FINISHED

# get the px of an azart in requested coord type (default is Kivy)
# calculation is 3 parts via azart to 1. xy_angs to 2. px to 3. KVpx
def AWIMmath_azarts_to_xyangs(AWIMtag_dictionary, azarts):
    if isinstance(azarts, list):
        azarts = np.asarray(azarts)

    input_shape = azarts.shape
    azarts = azarts.reshape(-1,2)

    center_azart = AWIMtag_dictionary['CenterAzArt']

    # find az_rels, convert to -180 < x <= 180, then abs value + direction matrix
    # also need az_rel compliment angle + store which are behind camera
    # then to radians
    simple_subtract = np.subtract(azarts[:,0], center_azart[0])
    big_angle_correction = np.where(simple_subtract > 0, -360, 360)
    az_rel = np.where(np.abs(simple_subtract) <= 180, simple_subtract, np.add(simple_subtract, big_angle_correction))
    az_rel_direction = np.where(az_rel < 0, -1, 1)
    az_rel_abs = np.abs(az_rel)
    az_rel_behind_observer = np.where(az_rel_abs <= 90, False, True) # true if point is behind observer - assume because camera pointed up very high past zenith or pointed very low below nether-zenith
    # Note: cannot allow az_rel_compliment (and therefore d2) to be negative because must simple_ang_totalsmallcircle be (-) if art_seg_ is (-), which is good.
    az_rel_compliment = np.where(az_rel_abs <= 90, np.subtract(90, az_rel_abs), np.subtract(az_rel_abs, 90)) # 0 to 90 angle from line perpendicular to az
    az_rel_compliment_rad = np.multiply(az_rel_compliment, math.pi/180) # (+) only, 0 to 90
    center_azart_rad = np.multiply(center_azart, math.pi/180)

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
    pt1_art_ = np.multiply(r2, np.sin(center_azart_rad[1])) # (-) for (-) cam_arts, which is good
    lower_half = np.where(art_seg_ < pt1_art_, True, False) # true if px is below middle of photo
    ang_smallcircle_fromhorizon = np.arctan(np.divide(art_seg_, d2)) # -90 to 90, (-) for (-) art_seg_, bc d2 always (+)
    # for yang, if in front, simple, but behind observer, the angle from must be subtracted from 180 or -180 because different angle meaning see photoshop diagram
    ang_totalsmallcircle = np.where(np.logical_not(az_rel_behind_observer), ang_smallcircle_fromhorizon, np.subtract(np.multiply(art_direction, math.pi), ang_smallcircle_fromhorizon))
    yang = np.subtract(ang_totalsmallcircle, center_azart_rad[1]) # simply subtract because |ang_totalsmallcircle| < 180 AND |center_azart[1]| < 90 AND if |ang_totalsmallcircle| > 90, then they are same sign

    xy_angs = np.zeros(input_shape)
    xy_angs[:,0] = np.multiply(xang, 180/math.pi)
    xy_angs[:,1] = np.multiply(yang, 180/math.pi)

    return xy_angs


# Part 2: xy_angs to pxs
def AWIMmath_xyangs_to_pxs(AWIMtag_dictionary, xy_angs):

    input_shape = xy_angs.shape

    xy_angs_direction = np.where(xy_angs < 0, -1, 1)
    xy_angs_abs = np.abs(xy_angs)

    if AWIMtag_dictionary['PixelMapType'] == '3d_degree_poly_fit_abs_from_center':
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
    px_models_df = AWIMtag_dictionary['PixelModels']
    x_px_predict_coeff = px_models_df['']
    pxs[:,0] = np.dot(xy_angs_poly, self.x_px_predict_coeff)
    pxs[:,1] = np.dot(xy_angs_poly, self.y_px_predict_coeff)

    pxs = np.multiply(pxs, xy_angs_direction)

    return pxs


# Part 3. pxs to coord_type
def px_coord_format(coord_type):
    if coord_type == 'KVpx':
        pxs[:,0] = pxs[:,0] + self.center_KVpx[0]
        pxs[:,1] = pxs[:,1] + self.center_KVpx[1]


def AWIMmath_pxs_to_xyangs(AWIMtag_dictionary, pxs):
    if isinstance(pxs, (list, tuple)): # models require numpy arrays
        pxs = np.asarray(pxs)

    input_shape = pxs.shape
    angs_direction = np.where(pxs < 0, -1, 1) # models are positive values only. Save sign. Same sign for xyangs

    pxs = np.abs(pxs).reshape(-1,2)

    if AWIMtag_dictionary['PixelMapType'] == '3d_degree_poly_fit_abs_from_center':
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

    xyangs = np.zeros(pxs.shape)
    xyangs[:,0] = np.dot(pxs_poly, self.xang_predict_coeff)
    xyangs[:,1] = np.dot(pxs_poly, self.yang_predict_coeff)

    xyangs_pretty = np.multiply(xyangs.reshape(input_shape), angs_direction)

    return xyangs_pretty, angs_direction


def AWIMmath_xyangs_to_azarts(AWIMtag_dictionary, xy_angs):

    # prepare to convert xyangs to azarts. Already have angs_direction from above. abs of xangs only, keep negative yangs
    angs_direction = angs_direction.reshape(-1,2)
    xyangs = xyangs_pretty.reshape(-1,2)
    xyangs[:,0] = np.abs(xyangs[:,0])
    xyangs = np.multiply(xyangs, math.pi/180)
    center_azart_rad = np.multiply(self.center_azart, math.pi/180)

    # see photoshop diagram of sphere, circles, and triangles for variable names
    xang_compliment = np.subtract(math.pi/2, xyangs[:,0]) # always (+) because xang < 90
    d1 = 1*np.cos(xang_compliment) # always (+)
    r2 = 1*np.sin(xang_compliment) # always (+)
    ang_totalsmallcircle = np.add(center_azart_rad[1], xyangs[:,1]) # -180 to 180
    d2_ = np.multiply(np.cos(ang_totalsmallcircle), r2) # (-) for ang_totalsmallcircle > 90 or < -90, meaning px behind observer
    art_seg_ = np.multiply(np.sin(ang_totalsmallcircle), r2) # (-) for (-) ang_totalsmallcircle
    arts = np.arcsin(art_seg_) # (-) for (-) art_seg_
    az_rel = np.subtract(math.pi/2, np.arctan(np.divide(d2_, d1))) # d2 (-) for px behind observer and therefore az_rel > 90 because will subtract (-) atan
    az_rel = np.multiply(az_rel, angs_direction[:,0])
    azs = np.mod(np.add(center_azart_rad[0], az_rel), 2*math.pi)

    azarts = np.zeros(xyangs.shape)
    azarts[:,0] = np.multiply(azs, 180/math.pi)
    azarts[:,1] = np.multiply(arts, 180/math.pi)

    azarts = azarts.reshape(input_shape)

    return azarts


# TODO
def get_celestial_azart(celestial_object):
    img_astropy_time = Time(capture_moment)
    img_astropy_location = EarthLocation(lat=earth_latlng[0]*u.deg, lon=earth_latlng[1]*u.deg)
    img_astropy_frame = AltAz(obstime=img_astropy_time, location=img_astropy_location)
    if celestial_object == 'sun':
        object_altaz = get_sun(img_astropy_time).transform_to(img_astropy_frame)
        known_azart = [object_altaz.az.degree, object_altaz.alt.degree]


# TODO this is copied from AWIM, not sure I should use it...
def px_azart_from_known_px_azart(AWIMtag_dictionary):
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

    if isinstance(known_azart_in, (list, np.ndarray)):
        known_azart = known_azart_in
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
    obj_alt_rad = known_azart[1] * math.pi/180
    d1 = 1*math.cos(math.pi/2 - xang_rel_rad)
    r2 = 1*math.sin(math.pi/2 - xang_rel_rad)
    alt_seg = 1*math.sin(obj_alt_rad)
    alt_ref_rad = math.asin(alt_seg / r2) - y_direction*yang_rel_rad
    d2 = r2 * math.cos(alt_ref_rad + yang_rel_rad)
    az_rel_rad = math.pi/2 - math.atan(d2 / d1)

    az_rel = az_rel_rad * 180/math.pi
    alt_ref = alt_ref_rad  * 180/math.pi

    # subtract az_rel because az_rel direction is opposite the camera reference
    azart_ref = [known_azart[0] - az_rel*x_direction, alt_ref]

    return azart_ref
