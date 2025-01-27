import os, shutil
import pyexiv2
import PIL


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


# or possibly save the metadata as XMP in order to match existing standard
def save_metadata_as_text(metadata_source_path):
	pass


def png_text_reader(image_filename):
    png_file_1 = PIL.Image.open(image_filename)
    png_text_dictionary = png_file_1.text

    for key, value in png_text_dictionary.items():
        print(key, ':', png_text_dictionary[key])

    return png_text_dictionary


def generate_png_with_awim_tag(current_image, rotate_degrees, awim_dictionary):
	# create the info object, add the awim data to the info object, save the png with the info object 
    png_data_container = PIL.PngImagePlugin.PngInfo()

    for key, value in awim_dictionary.items():
        png_data_container.add_text(key, value)
    
    save_filename_string = os.path.splitext(current_image.filename)[0] + ' - awim.png'
    current_image = current_image.rotate(angle=rotate_degrees, expand=True) # rotates CW
    current_image.save(save_filename_string, 'PNG', pnginfo=png_data_container)