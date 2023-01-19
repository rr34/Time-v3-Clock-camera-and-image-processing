from tkinter import Tk, PhotoImage, StringVar, Label, Button
from os import listdir, rename
from os.path import join, basename, splitext
import datetime
from pytz import timezone
import actions, awimlib, metadata_tools

class AppWindow(Tk):
    def __init__(self):
        super().__init__()
        #---------------------------------------- App Initialization ------------------------------
        self.title('AstroWideImageMapper')
        # self.state('zoomed')
        #---------------------------------------- Load Icons --------------------------------------
        self.icons_dict = {}
        self.icon_sz = 100
        for filename in listdir(r'icons/'):
            if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.gif'):
                key = splitext(basename(filename))[0]
                icon_path = join(r'icons/', filename)
                tk_image = PhotoImage(file=icon_path)
                zoom = int(tk_image.width() / self.icon_sz)
                if zoom > 1:
                    tk_image = tk_image.subsample(zoom, zoom)
                self.icons_dict[key] = tk_image
        #---------------------------------------- App Variables -----------------------------------
        self.doing_now = StringVar(self)
        self.doing_now.set('Initial Value')
        #---------------------------------------- Layout ------------------------------------------
        self.doing_now_label = Label(self, textvariable=self.doing_now)
        self.doing_now_label.grid(row=10, column=10)
        self.file_sc = Button(self, text='where does this text go?', compound='left', image=self.icons_dict['key_f'], width=self.icon_sz, height=self.icon_sz)
        self.file_sc.grid(row=0, column=0)

#------------------------------------------- Procedural --------------------------------------
if __name__ == '__main__':
    awim_app = AppWindow()
    awim_app.mainloop()