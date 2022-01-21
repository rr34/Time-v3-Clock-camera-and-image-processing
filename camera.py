import numpy as np
import pandas as pd
from sklearn import linear_model
from functions import *

class CameraAim(object):
	"""
	Note: used the Astroplan Observer class as a template.

	A container class for information about the properties of a particular
	camera / lens / settings system, especially the direction of each pixel
	or manageable fraction of the pixels relative to the a reference pixel.
	CameraAim is specific to a camera with a specific lens attached and
	set to specific settings

	Note, pixel reference to numpy array reference: the pixels are referenced by the standard used by Photoshop etc,
	i.e. (x, y), top-left to bottom-right:
	e.g. the top-left pixel is (0, 0), top-right is (1919, 0), bottom-left is (0, 1079), bottom-right is (1919, 1079)
	Numpy arrays refence [row, column], which visually is like [y, x].
	Therefore, for an image that is 1920x1080 the array will be 1080 rows x 1920 columns.
	Image (0, 0) = array [0, 0]
	Image (0, 1919) = array [1919, 0]
	AltAz is altitude bottom to top then azimuth left to right. This makes readability difficult and
	viewing the reference pixel data kind of a mind-bend but sticking with the standard is necessary and
	makes the coding more straightforward than trying to "fix" this.
	TODO: make a visual version of reference pixel data entry to avoid errors. It is really a mind-bend!
	"""
	def __init__(self, camera_name=None, sensor_dimensions=None, lens_name=None,
				 zoom_factor=None, settings_notes=None, image_dimensions=None, orientation=None, aperture=None,
				 calibration_file=None, pixel_aim_file=None):
				 
		"""
        Parameters.
        ----------
		camera_name : the name of the camera. Can include some lens and settings notes for quick
			identification
			
		image_width : the width of the camera's sensor in pixels. This should be the maximum
			resolution of the camera.
		
		image_height : the height of the camera's sensor in pixels. This should be the maximum
			resolution of the camera. Landscape is the stadard so the height will normally be
			less than the width.
		
		lens_name, zoom, settings_notes : (optional) specifying these is optional, but each is
			critical to the image properties from a particular camera.
			
		aperture : (optional) the aperture should not affect the angles of the pixels, but is
			here for completeness as desired.
			
		calibration_file : (optional) file containing the calibration reference pixels data.
			
		"""
 
	
		self.camera_name = camera_name
		# all parameters / the rest are defined with a calibration CSV file and calibrate method
		"""
		self.sensor_width = sensor_dimensions[0]
		self.sensor_height = sensor_dimensions[1]
		self.max_x_ref = self.image_width-1
		self.max_y_ref = self.image_height-1
		self.center_x_ref = (self.max_x_ref / 2)
		self.center_y_ref = (self.max_y_ref / 2)
		"""


	# TODO def __repr__(self):
		"""
		readable representation of the '~camera.CameraAim' object
		"""

	
	# @classmethod ??
	# generates alt, az, pixel_delta data file called pixel_aim. resolution is 1920 wide regardless of camera resolution
	# designed to be done just one time per camera / lens / settings system
	def calibrate(self, calibration_file):

		calibration_df = pd.read_csv(calibration_file)

		# calibration data includes most of requirements to initialize the object.
		lens_name = calibration_df[calibration_df['type'] == 'lens_name']['misc'].iat[0]
		zoom_factor = calibration_df[calibration_df['type'] == 'zoom_factor']['misc'].iat[0]
		settings_notes = calibration_df[calibration_df['type'] == 'settings_notes']['misc'].iat[0]
		orientation = calibration_df[calibration_df['type'] == 'orientation']['misc'].iat[0]
		calibration_quadrant = calibration_df[calibration_df['type'] == 'quadrant']['misc'].iat[0]
		sensor_dimensions = [calibration_df[calibration_df['type'] == 'sensor_dimensions']['px_x'].iat[0], calibration_df[calibration_df['type'] == 'sensor_dimensions']['px_y'].iat[0]]
		max_sensor_index = np.subtract(sensor_dimensions, 1)
		image_dimensions = [calibration_df[calibration_df['type'] == 'image_dimensions']['px_x'].iat[0], calibration_df[calibration_df['type'] == 'image_dimensions']['px_y'].iat[0]]
		max_image_index = np.subtract(image_dimensions, 1)
		center_px = max_image_index / 2 # usually x.5, non-integer, bc even number of pixels means center is a boundary rather than a pixel
		if orientation == 'portrait':
			center_px = center_px[::-1]


		# generate accurate reference pixels from the calibration measurements
		# initialize variables
		section_count = 0
		target_top = 0
		target_right = 0
		target_bottom = 0
		target_left = 0
		aim_px = center_px
		grid_align_v_left = 0
		grid_align_v_right = 0
		grid_align_h_top = 0
		grid_align_h_bottom = 0
		reference_pixels = []
		reference_pixels.append([center_px[0], center_px[1], 0, 0])

		# for loop now row by row because data has to be processed in order
		for row in calibration_df[calibration_df['type'] == 'calibration'].iterrows():
			# prepare the for loop
			row = row[1] # row[0] is just the index. I want the data, don't care about the index
			do_what = row['misc']
			if do_what == 'end':
				break
			# the only numerical do_what is the angular size of target, so...
			elif 'targets_diameter' in do_what:
				targets_diameter_degrees = float(do_what.split()[1])
				section_count += 1 # means new calibration section, so increment, first is 1
				aim_target_pos_altaz_relcenter = [row['alt_rel'], row['az_rel']] # is the altaz of the target position AIMING for in the current section, relative to center
			elif do_what == 'target_top':
				target_top = [row['px_x'], row['px_y']]
			elif do_what == 'target_right':
				target_right = [row['px_x'], row['px_y']]
			elif do_what == 'target_bottom':
				target_bottom = [row['px_x'], row['px_y']]
			elif do_what == 'target_left':
				target_left = [row['px_x'], row['px_y']]
			elif do_what == 'grid_align_vertical_left':
				grid_align_v_left = [row['px_x'], row['px_y']]
				v_align_degrees_reltarget = [row['alt_rel'], row['az_rel']] # is the vertical align target rel to target
			elif do_what == 'grid_align_vertical_right':
				grid_align_v_right = [row['px_x'], row['px_y']]
			elif do_what == 'grid_align_horizontal_top':
				grid_align_h_top = [row['px_x'], row['px_y']]
				h_align_altaz_reltarget = [row['alt_rel'], row['az_rel']] # is the horizontal align target rel to target
			elif do_what == 'grid_align_horizontal_bottom':
				grid_align_h_bottom = [row['px_x'], row['px_y']]
			# if four sides of target have been set, find hit pixel, find altaz adjustment (= position of the target in the image), then re-zero variables
			if 0 not in (target_top, target_right, target_bottom, target_left):
				hit_px, target_pos_altaz_relaim = target_miss(target_top, target_right, target_bottom, target_left, targets_diameter_degrees, aim_px)
				target_top = 0
				target_right = 0
				target_bottom = 0
				target_left = 0
				print('shooting results: ', section_count, aim_px, hit_px, target_pos_altaz_relaim, '\n')
			# if vertical grid alignment grid point has been recorded, calculate grid rotation error, record point, re-zero variables
			if 0 not in (grid_align_v_left, grid_align_v_right):
				if aim_target_pos_altaz_relcenter[1] == 0: # can only use vertical align point if aligned on the azimuth axis
					grid_rotation_error_degreesCCW = grid_rotation_error(grid_align_v_top, grid_align_v_bottom, v_align_altaz_reltarget, targets_diameter_degrees, hit_px, orientation='vertical')

				# re-zero and move on to reference pixels
				grid_align_v_left = 0
				grid_align_v_right = 0

			if do_what == 'grid_point':
				theta, r = altaz_to_special_polar(float(row['alt_rel']), float(row['az_rel']))
				grid_rotation_adjust_alt_relgridpoint =  math.tan(grid_rotation_error_degreesCCW*math.pi/180) * r * math.sin(theta*math.pi/180) * -1
				alt = float(row['alt_rel']) + aim_target_pos_altaz_relcenter[0] + target_pos_altaz_relaim[0] + grid_rotation_adjust_alt
				grid_rotation_adjust_az =  math.tan(grid_rotation_error_degreesCCW*math.pi/180) * r * math.cos(theta*math.pi/180)
				az = float(row['az_rel']) + aim_target_pos_altaz_relcenter[1] + target_pos_altaz_relaim[1] + grid_rotation_adjust_az
				row_data = [float(row['px_x']), float(row['px_y']), alt, az]
				reference_pixels.append(row_data)

		reference_pixels = np.array(reference_pixels)
		print(reference_pixels)
		
		# create the model to predict the rest of the pixels' alt
		independent_vars = reference_pixels_df[['px_x', 'px_y']].to_numpy()
		dependent_vars = reference_pixels_df['alt'].to_numpy()
		alt_model = linear_model.LinearRegression()
		alt_model.fit(independent_vars, dependent_vars)

		# create the 3D array with meshgrids for first two layers
		pixel_aim = np.empty([self.image_height, self.image_width, 4]) # I want height rows of width columns each row.
		x = np.arange(self.image_width)
		y = np.arange(self.image_height)
		pixel_aim[:,:,0], pixel_aim[:,:,1] = np.meshgrid(x,y) # !!!index is [row#,column#] therefore [y,x] !!!
		pixel_aim[:,:,2] = alt_model.predict(pixel_aim[:,:,[0,1]].reshape(-1,2)).reshape(self.image_height, self.image_width)
		
		
		"""
		# create the "meshgrid" dataframe for the entire image, Note: this works ***slowly*** using dataframe and for loops.
		camera_aim_df = DataFrame(columns=self.dataframe_columns)
		print(camera_aim_df)
		for x in np.arange(step=100, stop=self.max_x_ref):
			for y in np.arange(step=100, stop=self.max_y_ref):
				camera_aim_df.loc[len(camera_aim_df.index)] = [x, y, np.nan, np.nan]


		# fill in the meshgrid dataframe with predicted values
		camera_aim_df['alt'] = alt_model.predict(camera_aim_df[['px_x', 'px_y']])
		"""
		
		return pixel_aim
		
		"""
		# first, if the data is for one quadrant, complete the reference pixel dataframe by mirroring
		if self.ref_data_type == "UL_quadrant":
			UR_quadrant_df = DataFrame(reference_pixels_df.values, columns=reference_pixels_df.columns)
			UR_quadrant_df['px_x'] = UR_quadrant_df['px_x'] * -1 + 2 * self.center_x_ref
			UR_quadrant_df['px_y'] = UR_quadrant_df['px_y']
			UR_quadrant_df['alt'] = UR_quadrant_df['alt']
			UR_quadrant_df['az'] = UR_quadrant_df['az'] * -1
			reference_pixels_df = reference_pixels_df.append(UR_quadrant_df, ignore_index=True)
			bottom_half_df = DataFrame(reference_pixels_df.values, columns=reference_pixels_df.columns)
			bottom_half_df['px_x'] = bottom_half_df['px_x']
			bottom_half_df['px_y'] = bottom_half_df['px_y'] * -1 + 2 * self.center_y_ref
			bottom_half_df['alt'] = bottom_half_df['alt'] * -1
			bottom_half_df['az'] = bottom_half_df['az']
			reference_pixels_df = reference_pixels_df.append(bottom_half_df, ignore_index=True)




		# start working with numpy arrays here, create the 3D array with meshgrids for first two layers

		return pixel_aim
	"""