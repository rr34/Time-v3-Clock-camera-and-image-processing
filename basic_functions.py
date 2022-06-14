import pickle
import os
import PIL
from PIL.ExifTags import TAGS, GPSTAGS
import datetime
import pytz


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
        UTC_datetime_str = exif_UTC.strftime(exif_datetime_format)

    return UTC_datetime_str, UTC_source