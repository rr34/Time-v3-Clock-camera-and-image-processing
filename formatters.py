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


# works 11 Oct 2022: todorename formats each individual exif value
def format_individual_exif_values(exif_value):
    if isinstance(exif_value, piltiff.IFDRational):
        if exif_value.denominator != 0:
            value_readable = exif_value.numerator / exif_value.denominator
        else:
            value_readable = float('nan')
    elif isinstance(exif_value, bytes):
        try:
            value_readable = exif_value.decode()
            if not value_readable.isprintable():
                value_readable = ''.join([str(ord(x))[1:] for x in value_readable])
        except (UnicodeDecodeError, AttributeError):
            print('something unexpected happened look here')
    elif isinstance(exif_value, str) and not exif_value.isprintable():
        value_readable = ''
        string_list = exif_value.split()
        for element in exif_value:
            if element.isprintable():
                value_readable += element
            else:
                pass
    else:
        value_readable = exif_value

    return value_readable


def format_GPS_latlng(exif_readable):
    lat_sign = lng_sign = GPS_latlng = GPS_alt = False

    if exif_readable.get('Exif.GPSInfo.GPSLatitudeRef') == 'N':
        lat_sign = 1
    elif exif_readable.get('Exif.GPSInfo.GPSLatitudeRef') == 'S':
        lat_sign = -1
    if exif_readable.get('Exif.GPSInfo.GPSLongitudeRef') == 'E':
        lng_sign = 1
    elif exif_readable.get('Exif.GPSInfo.GPSLongitudeRef') == 'W':
        lng_sign = -1
    if lat_sign and lng_sign and exif_readable.get('Exif.GPSInfo.GPSLatitude') and exif_readable.get('Exif.GPSInfo.GPSLongitude'):
        lat_dms_list = re.split(' |\/', exif_readable['Exif.GPSInfo.GPSLatitude'])
        lng_dms_list = re.split(' |\/', exif_readable['Exif.GPSInfo.GPSLongitude'])
        lat_deg = float(lat_dms_list[0]) / float(lat_dms_list[1])
        lat_min = float(lat_dms_list[2]) / float(lat_dms_list[3])
        lat_sec = float(lat_dms_list[4]) / float(lat_dms_list[5])
        lng_deg = float(lng_dms_list[0]) / float(lng_dms_list[1])
        lng_min = float(lng_dms_list[2]) / float(lng_dms_list[3])
        lng_sec = float(lng_dms_list[4]) / float(lng_dms_list[5])
        img_lat = lat_sign * lat_deg + lat_min/60 + lat_sec/3600
        img_lng = lng_sign * lng_deg + lng_min/60 + lng_sec/3600
        GPS_latlng = [img_lat, img_lng]

    if exif_readable.get('Exif.GPSInfo.GPSAltitudeRef') and exif_readable.get('Exif.GPSInfo.GPSAltitude'):
        if exif_readable['Exif.GPSInfo.GPSAltitudeRef'] == '0':
            GPS_alt_sign = 1
        else:
            print('Elevation is something unusual, probably less than zero like in Death Valley or something. Look here.')
            GPS_alt_sign = -1
        GPS_alt_rational = exif_readable['Exif.GPSInfo.GPSAltitude'].split('/')
        GPS_alt = float(GPS_alt_rational[0]) / float(GPS_alt_rational[1]) * GPS_alt_sign

    return GPS_latlng, GPS_alt


def dict_json_ready(any_dictionary):
    jsonable_dict = {}
    for key, value in any_dictionary.items():
        if isinstance(value, np.datetime64):
            jsonable_dict[key] = format_datetime(value, 'to string for AWIMtag')
        else:
            jsonable_dict[key] = value

        # if not isinstance(key, str):
        #     key = str(key)

    return jsonable_dict


# ----- unknown below this line -----
# output types: 1. 'string' is with new lines 2. 'dictionary' is comma-separated
def dictionary_to_readable_textfile(any_dictionary):
    dictionary_string_txtfile = ''

    for key, value in any_dictionary.items():
        if isinstance(value, (int, float)):
            value_string = json.dumps(value)
        elif isinstance(value, (list, tuple)):
            value_string = []
            for list_item in value:
                value_string.append(format_individual_exif_values(list_item))
            value_string = json.dumps(value_string)
        elif value is None:
            value_string = str(value)
        elif isinstance(value, np.ndarray):
            value_string = '\n' + json.dumps(value.tolist())
            value_string = value_string.replace('],', '],\n')
        elif isinstance(value, pd.DataFrame):
            value_string = value.to_csv(index_label='features')
        elif isinstance(value, dict):
            value_string = dictionary_to_readable_textfile(value)
        elif isinstance(value, str):
            value_string = value
        elif isinstance(value, (bytes, piltiff.IFDRational)):
            value_string = str(format_individual_exif_values(value))
        else:
            value_string = 'unexpected data type'

        if not isinstance(key, str):
            key = str(key)

        value_string = key + ': ' + value_string + '\n'
        dictionary_string_txtfile += value_string

    return dictionary_string_txtfile


def stringify_dictionary(any_dictionary):
    dictionary_ofstrings = {}

    for key, value in any_dictionary.items():
        if isinstance(value, (int, float)):
            value_string = json.dumps(value)
        elif isinstance(value, (list, tuple)):
            value_string = []
            for list_item in value:
                value_string.append(format_individual_exif_values(list_item))
            value_string = json.dumps(value_string)
        elif value is None:
            value_string = str(value)
        elif isinstance(value, np.ndarray):
            value_string = json.dumps(value.tolist())
        elif isinstance(value, pd.DataFrame):
            value_string = value.to_csv()
        elif isinstance(value, dict):
            value_string = stringify_dictionary(value)
        elif isinstance(value, str):
            value_string = value
        elif isinstance(value, (bytes, piltiff.IFDRational)):
            value_string = str(format_individual_exif_values(value))
        else:
            value_string = 'unexpected data type'

        if not isinstance(key, str):
            key = str(key)

        dictionary_ofstrings[key] = value_string

    dictionary_ofstrings_str = str(dictionary_ofstrings)

    return dictionary_ofstrings_str


def de_stringify_tag(AWIMtag_dictionary_string):
    AWIMstart = AWIMtag_dictionary_string.find("AWIMstart")
    AWIMend = AWIMtag_dictionary_string.find("AWIMend")
    AWIMtag_dictionary_string = AWIMtag_dictionary_string[AWIMstart+9:AWIMend]
    AWIMtag_dictionary_ofstrings = ast.literal_eval(AWIMtag_dictionary_string)
    AWIMtag_dictionary = {}
    AWIMtag_template = generate_empty_AWIMtag_dictionary()
    for key, value in AWIMtag_dictionary_ofstrings.items():
        if value is None or value == '':
            AWIMtag_dictionary[key] = None
        elif AWIMtag_template[key] == 'csv':
            AWIMtag_dictionary[key] = pd.read_csv(io.StringIO(value), index_col=0)
        elif isinstance(AWIMtag_template[key], (int, float, list, tuple)):
            AWIMtag_dictionary[key] = json.loads(value)
        elif isinstance(AWIMtag_template[key], np.datetime64):
            AWIMtag_dictionary[key] = format_datetime(value, 'from AWIM string')
        elif isinstance(AWIMtag_template[key], np.ndarray):
            AWIMtag_dictionary[key] = np.array(json.loads(value))
        else:
            AWIMtag_dictionary[key] = value

    return AWIMtag_dictionary