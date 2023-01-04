import os, shutil
import pyexiv2


def get_metadata(metadata_source_path):
    metadata_src_type = os.path.splitext(metadata_source_path)[-1]

    img_pyexiv2 = pyexiv2.Image(metadata_source_path)
    img_exif_readable = img_pyexiv2.read_exif()
    img_pyexiv2.close()

    if True: #check if there was exif data
        return img_exif_readable
    else:
        return False


def UserComment_append(filepath, text, overwrite_append='append', modify_copy='copy'):
	working_directory = os.path.dirname(filepath) + r'/'
	working_filename = os.path.splitext(os.path.basename(filepath))[0]
	working_extension = os.path.splitext(os.path.basename(filepath))[-1]
	existing_data = get_metadata(filepath)
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


def save_metadata_as_text(metadata_source_path):
	pass