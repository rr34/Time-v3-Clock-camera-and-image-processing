import os, shutil
import pyexiv2


def get_metadata(metadata_source_path, save_exif_text_file=False):
    metadata_src_type = os.path.splitext(metadata_source_path)[-1]

    img_pyexiv2 = pyexiv2.Image(metadata_source_path)
    img_exif_readable = img_pyexiv2.read_exif()
    img_pyexiv2.close()

    if save_exif_text_file:
        img_exif_readable_str = dictionary_to_readable_textfile(img_exif_readable, 'txtfile')
        savename = os.path.splitext(metadata_source_path)[0]
        with open(savename + '-exif_readable.txt', 'w') as f:
            f.write(img_exif_readable_str)

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
		existing_UserComment = existing_data.get['Exif.Photo.UserComment']
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

test_file = os.path.abspath(os.curdir) + r'/camera-lens-calibration-repository/' + 'test-file.jpg'
comment = 'did i write this here?'
UserComment_append(test_file, comment)

"""
# prepare to save by getting filename
if calimg_exif_readable.get('Make'):
	if cam_ID != '':
		cam_ID += ' - '
	cam_ID += calimg_exif_readable['Make']
if calimg_exif_readable.get('Model'):
	if cam_ID != '':
		cam_ID += ' '
	cam_ID += calimg_exif_readable['Model']
if calimg_exif_readable.get('LensModel'):
	if cam_ID != '':
		cam_ID += ' - '
	cam_ID += calimg_exif_readable['LensModel']
if calimg_exif_readable.get('FocalLength'):
	if cam_ID != '':
		cam_ID += ' at '
	cam_ID += str(calimg_exif_readable['FocalLength']) + 'mm'

if calimg_exif_readable.get('UserComment'):
	user_comment_existing = calimg_exif_readable['UserComment']
	print('Hey, there was already a user comment. That is unusual. Look at this comment: ' + user_comment_existing)
else:
	user_comment_existing = ''

calibration_image.save(savepath + cam_ID + '+cameraAWIMtag.jpg', exif=calimg_exif_raw)

commentstart = user_comment.find("commentstart")
commentend = user_comment.find("commentend")
comment_parsed = user_comment[commentstart+12:commentend]

return comment_parsed
"""