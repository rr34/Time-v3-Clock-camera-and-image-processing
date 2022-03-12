import tkinter
import pickle
from datetime import datetime, timedelta
from pytz import timezone
import PIL
import actions

global current_camera
global current_image
global GPS_info_present, img_latlng, img_elevation, image_capture_moment

def load_camera():
    global current_camera
    camera_filename = tkinter.filedialog.askopenfilename()
    camera_aim_pickle = open(camera_filename, 'rb')
    current_camera = pickle.load(camera_aim_pickle)
    camera_aim_pickle.close()

    current_camera_str.set(str(current_camera.camera_name))

def load_image():
    # load the image, display the exif data, prompt the user for appropriate input
    global current_image
    global GPS_info_present, img_latlng, img_elevation, image_capture_moment
    image_filename = tkinter.filedialog.askopenfilename()
    current_image = PIL.Image.open(image_filename)
    current_image_str.set(image_filename)

    GPS_info_present, img_latlng, img_elevation, image_capture_moment = actions.get_exif(current_image)
    info_str = 'EXIF Data\nLat / Long: [%.4f, %.4f]\nElevation: %.1f meters\nCapture Moment: %s' % (img_latlng[0], img_latlng[1], img_elevation, image_capture_moment.isoformat(timespec='seconds'))
    output1_str.set(info_str)

    entry1_label_str.set('Enter lat,long OR leave blank to accept EXIF lat, long')
    entry2_label_str.set('Elevation in meters OR leave blank to accept EXIF elevation')
    entry3_label_str.set('yyyy-mm-ddTHH:mm:ss OR leave blank to accept EXIF time')

def continue1():
    if azalt_source_var.get() == 'Pixel x, y of sun':
        entry4_label_str.set('Enter pixel x,y of sun')
        entry5_label_str.set('NR')
    elif azalt_source_var.get() == 'Any pixel x, y on horizon':
        entry4_label_str.set('Enter any pixel x,y on horizon')
        entry5_label_str.set('NR')
    elif azalt_source_var.get() == 'Pixel x,y on horizon, known azimuth':
        entry4_label_str.set('Pixel x, y on horizon')
        entry5_label_str.set('Azimuth')

def continue2():
    global current_camera, current_image
    global GPS_info_present, img_latlng, img_elevation, image_capture_moment

    if not GPS_info_present:
        entry1_str = entry1.get()
        img_latlng = [float(entry1_str.split(',')[0]), float(entry1_str.split(',')[1])]
        img_elevation = float(entry2.get())

    what_object = 'sun'
    center_ref = 'center'
    img_orientation = 'landscape'
    img_tilt = 0 # placeholder for image tilted. (+) image tilt is horizon tilted CW in the image, so left down, right up, i.e. camera was tilted CCW as viewing from behind.
    entry4_str = entry4.get()
    celestial_object_px = [float(entry4_str.split(',')[0]), float(entry4_str.split(',')[1])]

    azalt_ref = actions.azalt_ref_from_celestial(current_camera, current_image, image_capture_moment, img_latlng, center_ref, what_object, celestial_object_px, img_orientation, img_tilt)

    awim_dictionary = current_camera.awim_metadata_generate(current_camera, current_image, image_capture_moment, earth_latlng, center_ref, azalt_ref, img_orientation)


awim = tkinter.Tk()

awim.title('Astronomical Wide Image Mapper by Time v3 Technology')
awim.geometry('1200x800')

menu_bar = tkinter.Menu(awim)

file_menu = tkinter.Menu(menu_bar, tearoff=0)
file_menu.add_command(label='Load Camera', command=load_camera)
file_menu.add_command(label='Save', command=actions.do_nothing)
file_menu.add_separator()
file_menu.add_command(label='Exit', command=awim.quit)

camera_menu = tkinter.Menu(menu_bar, tearoff=0)
camera_menu.add_command(label='TODO Calibrate camera from calibration images', command=actions.do_nothing)
camera_menu.add_command(label='Generate camera aim file from calibration CSV', command=actions.generate_camera_aim_object)
camera_menu.add_command(label='Display camera aim object', command=actions.display_camera_aim_object)

image_menu = tkinter.Menu(menu_bar, tearoff=0)
image_menu.add_command(label='Load Image', command=load_image)
image_menu.add_command(label='Attach AWIM Data to image with Sun Position', command=lambda: awimmetadata.png_add_metadata(current_camera, current_image))
image_menu.add_command(label='Batch generate image astronomical data files', command=actions.do_nothing)

menu_bar.add_cascade(label='File', menu=file_menu, underline=0)
menu_bar.add_cascade(label='Camera', menu=camera_menu, underline=0)
menu_bar.add_cascade(label='Image', menu=image_menu, underline=0)

# Entries and outputs:
# Rows 0 and 1 are current camera and current image.
# Entries and outputs start on row 2
# Column 0 is entry labels and button.
# Column 1 is entries. 
# Column 2 is output labels
# Column 3 is outputs
row_height = 10
column_width = 40

current_camera_label = tkinter.Label(awim, text='Current camera: ')
current_camera_str = tkinter.StringVar()
current_camera_str.set('None selected yet.')
current_camera_identify = tkinter.Label(awim, textvariable=current_camera_str)
current_camera_label.grid(row=0, column=0, ipady=row_height, ipadx=column_width)
current_camera_identify.grid(row=0, column=1, ipady=row_height, ipadx=column_width)

current_image_label = tkinter.Label(awim, text='Current image: ')
current_image_str = tkinter.StringVar()
current_image_str.set('None selected yet.')
current_image_identify = tkinter.Label(awim, textvariable=current_image_str)
current_image_label.grid(row=1, column=0, ipady=row_height, ipadx=column_width)
current_image_identify.grid(row=1, column=1, ipady=row_height, ipadx=column_width)

output1_str = tkinter.StringVar()
output1_str.set('Output 1 label placeholder')
output1_label = tkinter.Label(awim, textvariable=output1_str)
output1_label.grid(row=2, column=1, ipady=row_height, ipadx=column_width)

entry1_label_str = tkinter.StringVar()
entry1_label_str.set('Entry 1 text placeholder')
entry1_label = tkinter.Label(awim, textvariable=entry1_label_str)
entry1 = tkinter.Entry(awim)
entry1_label.grid(row=3, column=0, ipadx=column_width)
entry1.grid(row=3, column=1, ipadx=column_width)

entry2_label_str = tkinter.StringVar()
entry2_label_str.set('Entry 2 text placeholder')
entry2_label = tkinter.Label(awim, textvariable=entry2_label_str)
entry2 = tkinter.Entry(awim)
entry2_label.grid(row=4, column=0, ipadx=column_width)
entry2.grid(row=4, column=1, ipadx=column_width)

entry3_label_str = tkinter.StringVar()
entry3_label_str.set('Entry 2 text placeholder')
entry3_label = tkinter.Label(awim, textvariable=entry3_label_str)
entry3 = tkinter.Entry(awim)
entry3_label.grid(row=5, column=0, ipadx=column_width)
entry3.grid(row=5, column=1, ipadx=column_width)

azalt_source_var = tkinter.StringVar(awim)
azalt_source_var.set('Pixel x, y of sun')
azalt_source_menu = tkinter.OptionMenu(awim, azalt_source_var, 'Pixel x, y of sun', 'Any pixel x, y on horizon', 'Pixel x, y on horizon, known azimuth')
azalt_source_menu.grid(row=6, column=0, columnspan=2, ipady=row_height, ipadx=column_width)

continue_button = tkinter.Button(awim, text='Continue', command=continue1)
continue_button.grid(row=7, column=0, columnspan=2, ipady=row_height, ipadx=column_width)

entry4_label_str = tkinter.StringVar()
entry4_label_str.set('Entry 4 text placeholder')
entry4_label = tkinter.Label(awim, textvariable=entry4_label_str)
entry4 = tkinter.Entry(awim)
entry4_label.grid(row=8, column=0, ipadx=column_width)
entry4.grid(row=8, column=1, ipadx=column_width)

entry5_label_str = tkinter.StringVar()
entry5_label_str.set('Entry 5 text placeholder')
entry5_label = tkinter.Label(awim, textvariable=entry5_label_str)
entry5 = tkinter.Entry(awim)
entry5_label.grid(row=9, column=0, ipadx=column_width)
entry5.grid(row=9, column=1, ipadx=column_width)

entry_button2 = tkinter.Button(awim, text='View AWIM Data', command=continue2)
entry_button2.grid(row=10, column=0, columnspan=2, ipadx=column_width)


awim.config(menu=menu_bar)

awim.mainloop()