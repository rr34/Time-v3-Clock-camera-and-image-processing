import tkinter
from functions import *
from actions import *

root = tkinter.Tk()
root.title('Wide Image Astronomical Data Processor by Time v3 Technology')

menu_bar = tkinter.Menu(root)
root.config(menu=menu_bar)

file_menu = tkinter.Menu(menu_bar, tearoff=0)
file_menu.add_command(label='Open', command=do_nothing)
file_menu.add_command(label='Save', command=do_nothing)
file_menu.add_command(label='Exit', command=root.destroy)

camera_menu = tkinter.Menu(menu_bar, tearoff=0)
camera_menu.add_command(label='Calibrate camera from calibration images', command=do_nothing)
camera_menu.add_command(label='Generate camera aim file from calibration', command=generate_camera_aim_object)

image_menu = tkinter.Menu(menu_bar, tearoff=0)
image_menu.add_command(label='Generate image astronomical data file', command=do_nothing)
image_menu.add_command(label='Batch generate image astronomical data files', command=do_nothing)

menu_bar.add_cascade(label='File', menu=file_menu, underline=0)
menu_bar.add_cascade(label='Camera', menu=camera_menu, underline=0)
menu_bar.add_cascade(label='Image', menu=image_menu, underline=0)

root.mainloop()