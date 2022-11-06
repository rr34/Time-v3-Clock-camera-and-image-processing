import subprocess

def read_UserComment(filepath):
    exiv2_args = "-pt --grep UserComment"
    try:
        user_comment = subprocess.check_output(["exiv2", "-pt", "--grep" , "UserComment", filepath])
    except subprocess.CalledProcessError as err:
        user_comment = err

    if isinstance(user_comment, bytes):
        user_comment = user_comment.decode()

    return user_comment

a = read_UserComment('LN101821.jpg')
print(a)