import numpy as np
from matplotlib import pyplot
import camera

cell_phone_1x = camera.CameraAim(camera_name='Slayer Mobile', image_width=1920, image_height=1080, source_file_location="camera_generic_example.csv", ref_data_type="UL_quadrant")

# initialize this instance of the CameraAim class external to the __init__ function
cell_phone_1x.pixel_aim = cell_phone_1x.angles_array_extrapolation()

cell_phone_1x.pixel_aim.to_csv('pixel_aim_results.csv', index=False)

"""
print(cell_phone_1x.__dict__, '\n\n')

print(cell_phone_1x.pixel_aim.shape, '\n\n')

print(cell_phone_1x.pixel_aim[0,0])
"""

pyplot.imshow(cell_phone_1x.pixel_aim)
pyplot.show()