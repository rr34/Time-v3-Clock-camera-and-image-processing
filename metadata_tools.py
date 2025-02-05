import os, shutil
import datetime, pytz
import pyexiv2
import PIL
import formatters


def get_metadata(image_file_path):
    metadata_src_type = os.path.splitext(image_file_path)[-1]
    # file_base = os.path.splitext(image_file_path)[0]

    if metadata_src_type.lower() == '.png':
        png_file_1 = PIL.Image.open(image_file_path)
        png_text_dictionary = png_file_1.text

        if 'XML:com.adobe.xmp' in png_text_dictionary: # check for correct key for XMP data
            AdobeXML = png_text_dictionary['XML:com.adobe.xmp']
            metadata_dict = formatters.AdobeXML_to_dict(AdobeXML)
            metadata_dict['Metadata Source'] = 'Adobe XML from PNG file'

            # txt_file_name = file_base + '.XML'
            # with open(txt_file_name, "w") as text_file:
            # 	text_file.write(AdobeXML)

        else: # this is for the few old awim png files I made where there was a separate text chunk for each parameter that results in a dictionary where the keys are awim
            metadata_dict = png_text_dictionary
            metadata_dict['Metadata Source'] = 'PNG text chunks, possibly from old awim tag method'

    elif metadata_src_type.lower() in ('.raw', '.arw', '.jpg', '.jpeg'):
        img_pyexiv2 = pyexiv2.Image(image_file_path)
        metadata_dict = img_pyexiv2.read_exif()
        img_pyexiv2.close()
        metadata_dict['Metadata Source'] = 'Exif using pyexiv2 from file of type ' + metadata_src_type

    else:
        metadata_dict = {}
        metadata_dict['Metadata Source'] = 'No metadata found.'

    return metadata_dict


def capture_moment_from_metadata(metadata_dict, tz_default=False):
    # todo: update the exif metadata getter to use the time formatter function
    exif_UTC = UTC_source = UTC_datetime_str = False
    exif_datetime_format = "%Y:%m:%d %H:%M:%S"

    # 1. use the GPS time tag if present and ignore the leap seconds
    if metadata_dict.get('Exif.Image.GPSTag') and metadata_dict.get('Exif.GPSInfo.GPSDateStamp') and metadata_dict.get('Exif.GPSInfo.GPSTimeStamp'):
        GPS_datestamp_list = metadata_dict['Exif.GPSInfo.GPSDateStamp'].split(':')
        GPS_timestamp = metadata_dict['Exif.GPSInfo.GPSTimeStamp']
        GPS_timestamp_list = re.split(' |\/', GPS_timestamp)
        GPSyear = int(GPS_datestamp_list[0])
        GPSmonth = int(GPS_datestamp_list[1])
        GPSday = int(GPS_datestamp_list[2])
        if len(GPS_timestamp_list) == 6 and GPS_timestamp_list[1] == '1' and GPS_timestamp_list[3] == '1' and GPS_timestamp_list[5] == '1':
            GPShour = int(GPS_timestamp_list[0])
            GPSminute = int(GPS_timestamp_list[2])
            GPSsecond = int(GPS_timestamp_list[4])
        exif_UTC = datetime.datetime(year = GPSyear, month = GPSmonth, day = GPSday, hour = GPShour, minute = GPSminute, second = GPSsecond)
        UTC_source = 'exif GPSDateStamp and exif GPSTimeStamp'
    
    # 2. use the datetimeoriginal with either UTCOffset tag or default timezone
    elif metadata_dict.get('Exif.Photo.DateTimeOriginal'):
        exif_datetime = metadata_dict['Exif.Photo.DateTimeOriginal']
        exif_datetime_object_naive = datetime.datetime.strptime(exif_datetime, exif_datetime_format)

        if  metadata_dict.get('Exif.Photo.OffsetTimeOriginal'):
            exif_UTC_offset_list = metadata_dict['Exif.Photo.OffsetTimeOriginal'].split(':')
            exif_UTC_offset_minutes = 60 * int(exif_UTC_offset_list[0]) + int(exif_UTC_offset_list[1])
            UTC_offset_timedelta = datetime.timedelta(minutes = exif_UTC_offset_minutes)
            exif_UTC = exif_datetime_object_naive - UTC_offset_timedelta
            UTC_source = 'exif DateTimeOriginal and exif OffsetTimeOriginal'
        else:
            if tz_default:
                exif_datetime_object_localized = tz_default.localize(exif_datetime_object_naive)
                exif_UTC = exif_datetime_object_localized.astimezone(pytz.utc)
                UTC_source = 'exif DateTimeOriginal adjusted with user-entered timezone'
            else:
                exif_UTC = exif_datetime_object_naive
                UTC_source = 'exif DateTimeOriginal not adjusted for timezone, probably not UTC'

    elif metadata_dict.get('origmeta DateTimeOriginal'):
         metadata_datetime = metadata_dict['origmeta DateTimeOriginal']
         UTC_datetime_str = formatters.format_datetime(metadata_datetime, 'to string for AWIMtag')
         UTC_source = 'PNG XML DateTimeOriginal, which includes the timezone offset used by the camera. NMR camera set to not use DST, so offset should match location standard time.'

    if exif_UTC:
        UTC_datetime_str = formatters.format_datetime(exif_UTC, 'to string for AWIMtag')

    return UTC_datetime_str, UTC_source


# ----- unknown below this line -----
def GPS_location_from_metadata(metadata_dict):
    if 'get from exif GPS':
        if exif_readable.get('GPSInfo'):
            location, location_altitude = format_GPS_latlng(exif_readable)
            if location:
                AWIMtag_dictionary['Location'] = [round(f, round_digits['lat long']) for f in location]
                AWIMtag_dictionary['Location Source'] = 'DSC exif GPS'
            else:
                AWIMtag_dictionary['Location Source'] = 'Attempted to get from exif GPS, but was not present or not complete.'

            if location_altitude:
                AWIMtag_dictionary['Location Altitude'] = round(location_altitude, round_digits['altitude'])
                AWIMtag_dictionary['Location Altitude Source'] = 'DSC exif GPS'
            else:
                AWIMtag_dictionary['Location Altitude Source'] = 'Attempted to get from exif GPS, but was not present or not complete.'
        else:
            AWIMtag_dictionary['Location Source'] = 'Attempted to get from exif GPS, but GPSInfo was not present at all in exif.'
            AWIMtag_dictionary['Location Altitude Source'] = 'Attempted to get from exif GPS, but GPSInfo was not present at all in exif.'
    else:
        print('Error')



def UserComment_append(filepath, text, overwrite_append='append', modify_copy='copy'):
	working_directory = os.path.dirname(filepath) + r'/'
	working_filename = os.path.splitext(os.path.basename(filepath))[0]
	working_extension = os.path.splitext(os.path.basename(filepath))[-1]
	existing_data = get_metadata(filepath) # need to recreate the get_metadata function to work with text sidecar files
	if existing_data.get('Exif.Photo.UserComment') and overwrite_append == 'append':
		existing_UserComment = existing_data['Exif.Photo.UserComment']
	else:
		existing_UserComment = ''
	
	if modify_copy == 'copy':
		new_file = working_directory + working_filename + '-copy' + working_extension
		shutil.copy2(filepath, new_file)
	else:
		new_file = filepath

	file_pyexiv2 = pyexiv2.Image(filepath)
	modify_dictionary = {}
	modify_dictionary['Exif.Photo.UserComment'] = existing_UserComment + text
	file_pyexiv2.modify_exif(modify_dictionary)

	return new_file


def generate_png_with_awim_tag(current_image, rotate_degrees, awim_dictionary):
	# create the info object, add the awim data to the info object, save the png with the info object 
    png_data_container = PIL.PngImagePlugin.PngInfo()

    for key, value in awim_dictionary.items():
        png_data_container.add_text(key, value)
    
    save_filename_string = os.path.splitext(current_image.filename)[0] + ' - awim.png'
    current_image = current_image.rotate(angle=rotate_degrees, expand=True) # rotates CW
    current_image.save(save_filename_string, 'PNG', pnginfo=png_data_container)