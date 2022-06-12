import os
import pickle
import datetime, pytz
from PIL.ExifTags import TAGS, GPSTAGS

def exif_to_pickle(current_image):
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

        with open((os.path.splitext(current_image.filename)[0] + '.exifpickle'), 'wb') as current_image_pickle:
            pickle.dump(exifs, current_image_pickle, 5)

        return True
    else:
        return False


def UTC_from_exif(current_image, tz_default):
    with open((os.path.splitext(current_image.filename)[0] + '.exifpickle'), 'rb') as current_image_pickle:
        exif_readable = pickle.load(current_image_pickle)[1]

    # 1. use the GPS time tag if present and ignore the leap seconds
    if exif_readable['GPSInfo'].get('GPSDateStamp') and exif_readable['GPSInfo'].get('GPSTimeStamp'):
        GPS_datestamp = exif_readable['GPSInfo']['GPSDateStamp']
        GPS_timestamp = exif_readable['GPSInfo']['GPSTimeStamp']
        GPSyear = int(GPS_datestamp.split(':')[0])
        GPSmonth = int(GPS_datestamp.split(':')[1])
        GPSday = int(GPS_datestamp.split(':')[2])
        GPShour = int(GPS_timestamp[0])
        GPSminute = int(GPS_timestamp[1])
        GPSsecond = int(GPS_timestamp[2])
        exif_UTC = datetime.datetime(year = GPSyear, month = GPSmonth, day = GPSday, hour = GPShour, minute = GPSminute, second = GPSsecond)
    # 2. use the datetimeoriginal with either UTCOffset tag or default timezone
    elif exif_readable.get('DateTimeOriginal'):
        exif_datetime = exif_readable['DateTimeOriginal']
        exif_datetime_format = "%Y:%m:%d %H:%M:%S"
        exif_datetime_object_naive = datetime.datetime.strptime(exif_datetime, exif_datetime_format)

        if  exif_readable.get('OffsetTimeOriginal'):
            exif_UTC_offset_minutes = 60 * int(exif_readable['OffsetTimeOriginal'].split(':')[0]) + int(exif_readable['OffsetTimeOriginal'].split(':')[1])
            UTC_offset_timedelta = datetime.timedelta(minutes = exif_UTC_offset_minutes)
            exif_UTC = exif_datetime_object_naive - UTC_offset_timedelta
        else:
            exif_datetime_object_localized = tz_default.localize(exif_datetime_object_naive)
            exif_UTC = exif_datetime_object_localized.astimezone(pytz.utc)


    year = int(exif_date.split(':')[0])
    month = int(exif_date.split(':')[1])
    day = int(exif_date.split(':')[2])
    hour = int(exif_time.split(':')[0])
    minute = int(exif_time.split(':')[1])
    second = int(exif_time.split(':')[2])

