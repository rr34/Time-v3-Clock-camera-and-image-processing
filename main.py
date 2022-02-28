import tkinter
import pickle
from datetime import datetime
import functions
import actions

global current_camera

def load_camera():
    global current_camera
    camera_filename = tkinter.filedialog.askopenfilename()
    camera_aim_pickle = open(camera_filename, 'rb')
    current_camera = pickle.load(camera_aim_pickle)
    camera_aim_pickle.close()

    current_camera_str.set(str(current_camera.camera_name))

def button1_clicked():
    x_px = float(entry1.get())
    y_px = float(entry2.get())
    px_entry = [x_px, y_px]

    altaz = current_camera.cam_px_to_altaz(px_entry)
    output_str.set(str(altaz))

def button2_clicked():
    global current_camera
    image_filename = 'slayer_cal_image_1080p.png'
    img_capture_moment = datetime.now().isoformat(timespec='seconds')
    earth_latlng = [40.0, -83.0]
    center_ref = [959.5, 539.5]
    azalt_ref = [180.0, 10.0]

    current_camera.awim_metadata_to_image(image_filename=image_filename, date_gregorian_ns_time_utc = img_capture_moment, earth_latlng=earth_latlng, center_ref=center_ref, azalt_ref=azalt_ref)
        
awim = tkinter.Tk()

awim.title('Astronomical Wide Image Mapper by Time v3 Technology')
awim.geometry('800x1200')

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
# camera_menu.add_command(label='Convert a single pixel to altaz', command=single_px_to_altaz)

image_menu = tkinter.Menu(menu_bar, tearoff=0)
image_menu.add_command(label='Generate image astronomical data file', command=lambda: awimmetadata.png_add_metadata(current_camera))
image_menu.add_command(label='Batch generate image astronomical data files', command=actions.do_nothing)

menu_bar.add_cascade(label='File', menu=file_menu, underline=0)
menu_bar.add_cascade(label='Camera', menu=camera_menu, underline=0)
menu_bar.add_cascade(label='Image', menu=image_menu, underline=0)

current_camera_str = tkinter.StringVar()
current_camera_str.set('None yet')
output_str = tkinter.StringVar()
output_str.set('Nothing yet')

entry1_label = tkinter.Label(awim, text='Enter x coordinate, center is zero: ')
entry1_label.grid(row=0, column=0)
entry1 = tkinter.Entry(awim)
entry1.grid(row=0, column=1)
entry2_label = tkinter.Label(awim, text='Enter y coordinate, center is zero: ')
entry2_label.grid(row=1, column=0)
entry2 = tkinter.Entry(awim)
entry2.grid(row=1, column=1)
entry_button1 = tkinter.Button(awim, text='Enter', command=button1_clicked)
entry_button1.grid(row=2, column=0)
output_label = tkinter.Label(awim, textvariable=output_str)
output_label.grid(row=3, column=0)
current_camera_label = tkinter.Label(awim, textvariable=current_camera_str)
current_camera_label.grid(row=0, column=3)
entry_button2 = tkinter.Button(awim, text='Attach AWIM Data', command=button2_clicked)
entry_button2.grid(row=0, column=4)

awim.config(menu=menu_bar)

awim.mainloop()