import tkinter
import datetime
from pytz import timezone
import actions, awimlib, metadata_tools


class FrameOne(tkinter.Frame):
    def __init__(self, the_app):
        tkinter.Frame.__init__(self, the_app)

        self.bg = 'red'


class MenuBar(tkinter.Menu):
    def __init__(self, the_app):
        tkinter.Menu.__init__(self, the_app)

        filemenu = tkinter.Menu(self, tearoff=False)
        filemenu.add_command(label='Exit', command=self.exitapp)
        self.add_cascade(label='File', underline=0, menu=filemenu)

        camera_menu = tkinter.Menu(self, tearoff=False)
        camera_menu.add_command(label='Generate camera AWIM file from calibration CSV', command=self.blue_frame)
        camera_menu.add_command(label='Display camera AWIM object file', command=self.exitapp)
        camera_menu.add_command(label='TODO Calibrate camera from calibration images', command=self.exitapp)
        self.add_cascade(label='Camera Menu', underline=0, menu=camera_menu)

    def exitapp(self):
        self.quit()

    def blue_frame(self):
        frame = FrameOne(self)
        frame.pack(fill='both', expand=1)


class AWIMapp(tkinter.Tk):
    def __init__(self, title, geometry):
        super().__init__() # this is calling the __init__ from its parent, tkinter.Tk

        self.title(title)
        self.geometry(geometry)
        menu_bar = MenuBar(self)
        self.config(menu=menu_bar)

        frame_one = FrameOne(self)

        button = tkinter.Button(self, text='test button', command=menu_bar.exitapp)
        button.grid(row=0, column=0)


if __name__ == '__main__':
    AWIMapp = AWIMapp('AstroWideImageMapper', '1200x800')
    AWIMapp.mainloop()