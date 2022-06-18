import tkinter
import pickle
import datetime
from pytz import timezone
import actions, astroimage

# I know I'm not supposed to use globals. They are not referenced outside this file.
# If I convert this app to object oriented, they will become self._____
global current_camera
global rotate_degrees, GPS_info_present, img_latlng, img_elevation, image_capture_moment, awim_dictionary_in


def load_camera():
    global current_camera
    camera_filename = tkinter.filedialog.askopenfilename()
    camera_aim_pickle = open(camera_filename, 'rb')
    current_camera = pickle.load(camera_aim_pickle)
    camera_aim_pickle.close()

    current_camera_str.set(str(current_camera.camera_name))


def process_image():
    global current_camera, current_image
    global rotate_degrees, exif_present, GPS_info_present, img_latlng, img_elevation, image_capture_moment, tz_default
    camera_path = None
    src_img_path = tkinter.filedialog.askopenfilename()
    metadata_source_path = src_img_path
    current_image_str.set(src_img_path)

    # hard coded. TODO make user-input
    tz_default = timezone('US/Eastern')
    center_ref = 'center'
    img_orientation = 'landscape'
    img_tilt = 0 # placeholder for image tilted. (+) image tilt is horizon tilted CW in the image, so left down, right up, i.e. camera was tilted CCW as viewing from behind. Which axis? I think should be around the camera axis.


    actions.generate_image_with_AWIM_tag(src_img_path, metadata_source_path, \
                                        tz_default, center_ref, current_camera, img_orientation, img_tilt)

def continue1():
    if azart_source_var.get() == 'Pixel x,y of sun':
        entry4_label_str.set('Enter pixel x,y of sun')
        entry5_label_str.set('NR')
    elif azart_source_var.get() == 'Pixel x,y on horizon, with known azimuth to pixel':
        entry4_label_str.set('Pixel x,y on horizon')
        entry5_label_str.set('Reference Azimuth')
    elif azart_source_var.get() == 'Manual Az,art':
        entry4_label_str.set('Az,art')
        entry5_label_str.set('NR')


def continue2():
    global current_camera, current_image
    global rotate_degrees, exif_present, GPS_info_present, img_latlng, img_elevation, image_capture_moment, awim_dictionary_in

    if not exif_present:
        image_capture_moment = datetime.datetime.fromisoformat(entry3.get() + '+00:00')

    if not GPS_info_present:
        entry1_str = entry1.get()
        img_latlng = [float(entry1_str.split(',')[0]), float(entry1_str.split(',')[1])]
        img_elevation = float(entry2.get())

    img_orientation = 'landscape'
    center_ref = 'center'
    img_tilt = 0 # placeholder for image tilted. (+) image tilt is horizon tilted CW in the image, so left down, right up, i.e. camera was tilted CCW as viewing from behind. Which axis? I think should be around the camera axis.

    if azart_source_var.get() == 'Manual Az,Art':
        azart_str = entry4.get()
        azart_ref = [float(azart_str.split(',')[0]), float(azart_str.split(',')[1])]
    elif azart_source_var.get() == 'Pixel x,y on horizon, with known azimuth to pixel':
        px_coord_str = entry4.get()
        azart_horizon = [float(entry5.get()), 0]
        known_pt_px = [float(px_coord_str.split(',')[0]), float(px_coord_str.split(',')[1])]
        azart_ref = actions.azart_ref_from_known_px(current_camera, current_image, image_capture_moment, img_latlng, center_ref, azart_horizon, known_pt_px, img_orientation, img_tilt)
    elif azart_source_var.get() == 'Pixel x,y of sun':
        known_pt_px = 'sun'
        px_coord_str = entry4.get()
        celestial_object_px = [float(px_coord_str.split(',')[0]), float(px_coord_str.split(',')[1])]
        azart_ref = actions.azart_ref_from_known_px(current_camera, current_image, image_capture_moment, img_latlng, center_ref, known_pt_px, celestial_object_px, img_orientation, img_tilt)

    awim_dictionary_in = current_camera.generate_xyang_pixel_models(current_image, image_capture_moment, img_latlng, center_ref, azart_ref, img_orientation, img_tilt)
    
    awim_dictionary_str = ''
    for item in awim_dictionary_in:
        awim_dictionary_str += item + ': ' + awim_dictionary_in[item] + '\n'
    output2_str.set('Center AzArt: ' + str(awim_dictionary_in['Center AzArt']) + '\nsee file code output dump/image awim data.txt for full AWIM tag')
    with open(r'code output dump folder/image awim data.txt', 'w') as f:
        f.write(awim_dictionary_str)


def continue3():
    global current_camera, current_image
    global rotate_degrees, exif_present, GPS_info_present, img_latlng, img_elevation, image_capture_moment, awim_dictionary_in

    actions.generate_png_with_awim_tag(current_image, rotate_degrees, awim_dictionary_in)


def png_read():

    png_filename = tkinter.filedialog.askopenfilename()
    awim_dictionary = actions.png_text_reader(png_filename)
    clock_image_data_obj = astroimage.ImageAWIMData(awim_dictionary)


AWIMtkapp = tkinter.Tk()

AWIMtkapp.title('Astronomical Wide Image Mapper by Time v3 Technology')
AWIMtkapp.geometry('1200x800')

menu_bar = tkinter.Menu(AWIMtkapp)

file_menu = tkinter.Menu(menu_bar, tearoff=0)
file_menu.add_command(label='Load Camera', command=load_camera)
file_menu.add_command(label='Load Image', command=process_image)
file_menu.add_command(label='Save', command=actions.do_nothing)
file_menu.add_separator()
file_menu.add_command(label='Exit', command=AWIMtkapp.quit)

camera_menu = tkinter.Menu(menu_bar, tearoff=0)
camera_menu.add_command(label='Generate camera AWIM file from calibration CSV', command=actions.generate_save_camera_AWIM)
camera_menu.add_command(label='Display camera AWIM object file', command=actions.display_camera_AWIM_object)
camera_menu.add_command(label='TODO Calibrate camera from calibration images', command=actions.do_nothing)

image_menu = tkinter.Menu(menu_bar, tearoff=0)
image_menu.add_command(label='Load PNG with AWIM Data', command=png_read)
image_menu.add_command(label='TODO Batch generate AWIM-tagged images from directory', command=actions.do_nothing)

menu_bar.add_cascade(label='File', menu=file_menu, underline=0)
menu_bar.add_cascade(label='Camera', menu=camera_menu, underline=0)
menu_bar.add_cascade(label='Image', menu=image_menu, underline=0)

row_height = 10
column_width = 40

current_camera_label = tkinter.Label(AWIMtkapp, text='Current camera: ')
current_camera_str = tkinter.StringVar()
current_camera_str.set('None selected yet.')
current_camera_identify = tkinter.Label(AWIMtkapp, textvariable=current_camera_str)
current_camera_label.grid(row=0, column=0, ipady=row_height, ipadx=column_width)
current_camera_identify.grid(row=0, column=1, ipady=row_height, ipadx=column_width)

current_image_label = tkinter.Label(AWIMtkapp, text='Current image: ')
current_image_str = tkinter.StringVar()
current_image_str.set('None selected yet.')
current_image_identify = tkinter.Label(AWIMtkapp, textvariable=current_image_str)
current_image_label.grid(row=1, column=0, ipady=row_height, ipadx=column_width)
current_image_identify.grid(row=1, column=1, ipady=row_height, ipadx=column_width)

output1_str = tkinter.StringVar()
output1_str.set('Output 1 label placeholder')
output1_label = tkinter.Label(AWIMtkapp, textvariable=output1_str)
output1_label.grid(row=2, column=1, ipady=row_height, ipadx=column_width)

entry1_label_str = tkinter.StringVar()
entry1_label_str.set('Entry 1 text placeholder')
entry1_label = tkinter.Label(AWIMtkapp, textvariable=entry1_label_str)
entry1 = tkinter.Entry(AWIMtkapp)
entry1_label.grid(row=3, column=0, ipadx=column_width)
entry1.grid(row=3, column=1, ipadx=column_width)

entry2_label_str = tkinter.StringVar()
entry2_label_str.set('Entry 2 text placeholder')
entry2_label = tkinter.Label(AWIMtkapp, textvariable=entry2_label_str)
entry2 = tkinter.Entry(AWIMtkapp)
entry2_label.grid(row=4, column=0, ipadx=column_width)
entry2.grid(row=4, column=1, ipadx=column_width)

entry3_label_str = tkinter.StringVar()
entry3_label_str.set('Entry 2 text placeholder')
entry3_label = tkinter.Label(AWIMtkapp, textvariable=entry3_label_str)
entry3 = tkinter.Entry(AWIMtkapp)
entry3_label.grid(row=5, column=0, ipadx=column_width)
entry3.grid(row=5, column=1, ipadx=column_width)

azart_source_var = tkinter.StringVar(AWIMtkapp)
azart_source_var.set('Pixel x,y of sun')
azart_source_menu = tkinter.OptionMenu(AWIMtkapp, azart_source_var, 'Pixel x,y of sun', 'Pixel x,y on horizon, with known azimuth to pixel', 'Manual Az,Art')
azart_source_menu.grid(row=6, column=0, columnspan=2, ipady=row_height, ipadx=column_width)

continue_button = tkinter.Button(AWIMtkapp, text='Continue', command=continue1)
continue_button.grid(row=7, column=0, columnspan=2, ipady=row_height, ipadx=column_width)

entry4_label_str = tkinter.StringVar()
entry4_label_str.set('Entry 4 text placeholder')
entry4_label = tkinter.Label(AWIMtkapp, textvariable=entry4_label_str)
entry4 = tkinter.Entry(AWIMtkapp)
entry4_label.grid(row=8, column=0, ipadx=column_width)
entry4.grid(row=8, column=1, ipadx=column_width)

entry5_label_str = tkinter.StringVar()
entry5_label_str.set('Entry 5 text placeholder')
entry5_label = tkinter.Label(AWIMtkapp, textvariable=entry5_label_str)
entry5 = tkinter.Entry(AWIMtkapp)
entry5_label.grid(row=9, column=0, ipadx=column_width)
entry5.grid(row=9, column=1, ipadx=column_width)

entry_button2 = tkinter.Button(AWIMtkapp, text='Show Data', command=continue2)
entry_button2.grid(row=10, column=0, columnspan=2, ipadx=column_width)

output2_str = tkinter.StringVar()
output2_str.set('Output 2 label placeholder')
output2_label = tkinter.Label(AWIMtkapp, textvariable=output2_str)
output2_label.grid(row=11, column=1, ipady=row_height, ipadx=column_width)

entry_button3 = tkinter.Button(AWIMtkapp, text='Generate PNG with Data', command=continue3)
entry_button3.grid(row=12, column=0, columnspan=2, ipadx=column_width)

AWIMtkapp.config(menu=menu_bar)

AWIMtkapp.mainloop()