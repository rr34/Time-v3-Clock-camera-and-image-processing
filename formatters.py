import datetime
import numpy as np
import re

# numpy datetime64 is International Atomic Time (TAI), not UTC, so it ignores leap seconds.
# numpy datetime64 uses astronomical year numbering, i.e. year 2BC = year -1, 1BC = year 0, 1AD = year 1
def format_datetimes(input_datetime, direction):
    # patterns for generating strings
    exif_datetime_format = "%Y:%m:%d %H:%M:%S" # directly from exif documentation
    numpy_datetime_format = "%Y-%m-%dT%H:%M:%S" # from numpy documentation, is timezone naive
    ISO8601_datetime_format = "%Y-%m-%dT%H:%M:%SZ" # ISO 8601
    filename_format = "%Y-%m-%dT%H%M%S" # filename

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