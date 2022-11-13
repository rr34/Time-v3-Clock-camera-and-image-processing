import subprocess

def append_UserComment(filepath, text_to_append):
    
    command1 = '-M'
    command2 = 'set Exif.Photo.UserComment '
    user_comment = 'commentstart' + 'i wrote a user comment here\nso you should be super-impressed' + 'commentend'
    success = True
    try:
        subprocess.call(["exiv2", command1+command2+user_comment, filepath])
    except subprocess.CalledProcessError:
        success = False

    return success

def read_UserComment(filepath):
    try:
        user_comment = subprocess.check_output(["exiv2", "-pt", "--grep" , "UserComment", filepath])
    except subprocess.CalledProcessError as err:
        user_comment = err

    if isinstance(user_comment, bytes):
        user_comment = user_comment.decode()

    commentstart = user_comment.find("commentstart")
    commentend = user_comment.find("commentend")
    comment_parsed = user_comment[commentstart+12:commentend]


    return comment_parsed

def read_exif_field(filepath):
    try:
        field_value = subprocess.check_output(["exiv2", "-K" , "Exif.Image.XResolution", filepath])
    except subprocess.CalledProcessError as err:
        field_value = err

    if isinstance(field_value, bytes):
        field_value_decoded = field_value.decode()

    return field_value

filepath = 'LN101821.jpg'

a = set_UserComment(filepath)

if a:
    b = read_UserComment(filepath)

c = read_exif_field(filepath)

print(c)
print(type(c))