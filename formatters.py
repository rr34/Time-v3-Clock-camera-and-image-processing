import datetime
import numpy as np
import re
from collections.abc import MutableMapping
import json
import PIL.TiffImagePlugin as piltiff
import xmltodict # xmltodict is maintained by an individual. Very widespread, but not a mainstream library.


def AWIMtag_rounding_digits():
    rounding_digits_dict = {}
    rounding_digits_dict['lat long'] = 6
    rounding_digits_dict['altitude'] = 1
    rounding_digits_dict['AGL'] = 2
    rounding_digits_dict['pixels'] = 1
    rounding_digits_dict['degrees'] = 2
    rounding_digits_dict['hourangle'] = 3

    return rounding_digits_dict


def round_AWIMtag(AWIMtag):
    round_digits_dict = AWIMtag_rounding_digits()
    for key, value in AWIMtag.items():
        round_this = False
        if key in ('awim Location Coordinates'):
            round_digits = round_digits_dict['lat long']
            round_this = True
        elif key in ('Location MSL', 'awim Location Terrain Elevation'):
            round_digits = round_digits_dict['altitude']
            round_this = True
        elif key in ('awim Location AGL'):
            round_digits = round_digits_dict['AGL']
            round_this = True
        elif key in ('awim Ref Pixel', 'awim Ref Image Size', 'awim Grid Pixels', 'awim Pixel Size Center Horizontal Vertical', 'awim Pixel Size Average Horizontal Vertical', 'awim TBLR Pixels'):
            round_digits = round_digits_dict['pixels']
            round_this = True
        elif key in ('awim Ref Pixel Azimuth Artifae', 'awim Grid Angles', 'awim Grid Azimuth Artifae', 'awim TBLR Angles', 'awim TBLR Azimuth Artifae'):
            round_digits = round_digits_dict['degrees']
            round_this = True
        elif key in ('awim Grid RA Dec','awim TBLR RA Dec'):
            round_digits = round_digits_dict['hourangle']
            round_this = True
            # todo: the declination should be rounded to a hundredth instead of thousandth, a problem for another day, something like this:
            # img_borders_RADecs[:,[0,2,4]] = img_borders_RADecs[:,[0,2,4]].round(round_digits['hourangle'])
	        # img_borders_RADecs[:,[1,3,5]] = img_borders_RADecs[:,[1,3,5]].round(round_digits['degrees'])

        if isinstance(value, list) and round_this:
            rounded = np.array(value).round(round_digits).tolist()
            AWIMtag[key] = rounded
        elif isinstance(value, (float)) and round_this:
            AWIMtag[key] = round(value, round_digits)

    return AWIMtag


# numpy datetime64 is International Atomic Time (TAI), not UTC, so it ignores leap seconds.
# numpy datetime64 uses astronomical year numbering, i.e. year 2BC = year -1, 1BC = year 0, 1AD = year 1
def format_datetime(input_datetime, direction):
    # patterns for generating strings
    exif_datetime_format = "%Y:%m:%d %H:%M:%S" # directly from exif documentation
    numpy_datetime_format = "%Y-%m-%dT%H:%M:%S" # from numpy documentation, is timezone naive
    ISO8601_datetime_format = "%Y-%m-%dT%H:%M:%SZ" # ISO 8601
    filename_format = "%Y-%m-%dT%H%M%SZ" # filename

    # regex patterns for recognizing strings
    ISO8601_pattern = r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}([+-]\d{2}:\d{2}|Z?)\s*'
    
    if direction == 'to string for exif':
        if isinstance(input_datetime, datetime.datetime):
            output = input_datetime.strftime(exif_datetime_format)
        elif isinstance(input_datetime, np.datetime64):
            pass # TODO convert this format, necessary?

    elif direction == 'to string for AWIMtag':
        if isinstance(input_datetime, datetime.datetime):
            output = input_datetime.strftime(ISO8601_datetime_format)
        elif isinstance(input_datetime, np.datetime64):
            output = str(np.datetime_as_string(input_datetime, unit='s')) + 'Z'
        elif isinstance(input_datetime, np.ndarray):
            output = np.datetime_as_string(input_datetime, unit='s')
            output = [str(dt)  + 'Z' for dt in output]
        elif isinstance(input_datetime, str):
            if re.match(ISO8601_pattern, input_datetime):
                output = str(np.datetime_as_string(np.datetime64(input_datetime), unit='s')) + 'Z'

    elif direction == 'to string for filename':
        if isinstance(input_datetime, datetime.datetime):
            output = input_datetime.strftime(filename_format)
        elif isinstance(input_datetime, np.datetime64):
            pass # TODO convert this format, necessary?

    elif direction == 'from string':
        if re.match(ISO8601_pattern, input_datetime):
            output = np.datetime64(input_datetime) # astropy uses numpy datetime64 primarily
        else:
            output = 'String was of unexpected format.'

    elif direction == 'from list of ISO 8601 strings':
        output = [np.datetime64(t) for t in input_datetime]

    elif direction == 'ISO 8601 string tz to Zulu':
        output = [str(np.datetime64(ts))+'Z' for ts in input_datetime]

    return output


def adjust_datetime_byseconds(datetime_str, adjustment):
    datetime_numpy1 = np.datetime64(datetime_str)
    timedelta_numpy = np.timedelta64(adjustment, 's')
    datetime_numpy2 = datetime_numpy1 - timedelta_numpy

    result_str = str(np.datetime_as_string(datetime_numpy2, unit='s')) + 'Z'

    return result_str


def round_to_string(numbers, type):
    rounding_digits_dict = {}
    rounding_digits_dict['lat long'] = 6
    rounding_digits_dict['azimuth'] = 2
    rounding_digits_dict['artifae'] = 1
    rounding_digits_dict['AGL'] = 2
    rounding_digits_dict['pixels'] = 1
    rounding_digits_dict['degrees'] = 2
    rounding_digits_dict['hourangle'] = 3

    round_digits = rounding_digits_dict[type]

    if type == 'azimuth':
        # output = [f'{int(azimuth):03d}' + f'{round(azimuth%1, 1)}'[1:] for azimuth in numbers]
        output = [f'{azimuth:6.2f}'.replace(' ', '0') for azimuth in numbers]
    elif round_digits == 1:
        output = [f'{number:.1f}' for number in numbers]
    elif round_digits == 2:
        output = [f'{number:.2f}' for number in numbers]
    elif round_digits == 3:
        output = [f'{number:.3f}' for number in numbers]
    elif round_digits == 6:
        output = [f'{number:.6f}' for number in numbers]

    return output


def flatten_dict(dictionary, parent_key='', separator='_'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


# this is specific to Adobe XML metadata as of 2025-01-31
def simplify_keys(metadata_dict):
    print('Keys before {}'.format(len(metadata_dict.keys())))

    # list of keys I want to be simple
    simple_keys = ['Make','Model','XResolution','YResolution','ResolutionUnit','ExifVersion','ExposureTime','ShutterSpeedValue','FNumber','ApertureValue','ExposureProgram','SensitivityType','RecommendedExposureIndex','BrightnessValue','ExposureBiasValue','MaxApertureValue','MeteringMode','LightSource','FocalLength','FileSource','SceneType','FocalLengthIn35mmFilm','CustomRendered','ExposureMode','WhiteBalance','SceneCaptureType','Contrast','Saturation','Sharpness','DigitalZoomRatio','FocalPlaneXResolution','FocalPlaneYResolution','FocalPlaneResolutionUnit','DateTimeOriginal']
    print('Simple keys to search {}'.format(len(simple_keys)))

    # make a new dictionary of the same information with the simple keys
    dict_withjust_new_keys = {}
    delete_after_loop = []
    for key in metadata_dict.keys():
        for simple_key in simple_keys:
            re_pattern = '(?<=tiff:){}$|(?<=exif:){}$'.format(simple_key, simple_key)
            if re.search(re_pattern, key):
                new_key = 'origmeta ' + simple_key
                if new_key not in dict_withjust_new_keys:
                    dict_withjust_new_keys[new_key] = metadata_dict[key]
                    delete_after_loop.append(key)
                else:
                    print('There is a duplicate match for ' + simple_key)

    # add the information to the original dictionary
    metadata_dict.update(dict_withjust_new_keys)
    print('Keys simplified {}'.format(len(dict_withjust_new_keys.keys())))

    # delete the information with the old keys, which is now duplicate
    for delete_this in delete_after_loop:
        del metadata_dict[delete_this]
    print('Keys after {}'.format(len(metadata_dict.keys())))

    return metadata_dict


def AdobeXML_to_dict(AdobeXML):
    metadata_dict = xmltodict.parse(AdobeXML)
    metadata_dict = flatten_dict(metadata_dict)
    metadata_dict = simplify_keys(metadata_dict)

    return metadata_dict


def parse_brightstar_text(brightstar_text):
    pass


# works 11 Oct 2022: todorename formats each individual exif value
def format_GPS_latlng(exif_dict):
    lat_sign = lng_sign = GPS_latlng = GPS_alt = False

    if exif_dict.get('Exif.GPSInfo.GPSLatitudeRef') == 'N':
        lat_sign = 1
    elif exif_dict.get('Exif.GPSInfo.GPSLatitudeRef') == 'S':
        lat_sign = -1
    if exif_dict.get('Exif.GPSInfo.GPSLongitudeRef') == 'E':
        lng_sign = 1
    elif exif_dict.get('Exif.GPSInfo.GPSLongitudeRef') == 'W':
        lng_sign = -1
    if lat_sign and lng_sign and exif_dict.get('Exif.GPSInfo.GPSLatitude') and exif_dict.get('Exif.GPSInfo.GPSLongitude'):
        lat_dms_list = re.split(' |\/', exif_dict['Exif.GPSInfo.GPSLatitude'])
        lng_dms_list = re.split(' |\/', exif_dict['Exif.GPSInfo.GPSLongitude'])
        lat_deg = float(lat_dms_list[0]) / float(lat_dms_list[1])
        lat_min = float(lat_dms_list[2]) / float(lat_dms_list[3])
        lat_sec = float(lat_dms_list[4]) / float(lat_dms_list[5])
        lng_deg = float(lng_dms_list[0]) / float(lng_dms_list[1])
        lng_min = float(lng_dms_list[2]) / float(lng_dms_list[3])
        lng_sec = float(lng_dms_list[4]) / float(lng_dms_list[5])
        img_lat = lat_sign * lat_deg + lat_min/60 + lat_sec/3600
        img_lng = lng_sign * lng_deg + lng_min/60 + lng_sec/3600
        GPS_latlng = [img_lat, img_lng]

    if exif_dict.get('Exif.GPSInfo.GPSAltitudeRef') and exif_dict.get('Exif.GPSInfo.GPSAltitude'):
        if exif_dict['Exif.GPSInfo.GPSAltitudeRef'] == '0':
            GPS_alt_sign = 1
        else:
            print('Elevation is something unusual, probably less than zero like in Death Valley or something. Look here.')
            GPS_alt_sign = -1
        GPS_alt_rational = exif_dict['Exif.GPSInfo.GPSAltitude'].split('/')
        GPS_alt = float(GPS_alt_rational[0]) / float(GPS_alt_rational[1]) * GPS_alt_sign

    return GPS_latlng, GPS_alt