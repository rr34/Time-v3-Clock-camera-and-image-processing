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
				 zoom_factor=None, settings_notes=None, image_dimensions=None, aperture=None,
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
		self.lens_name = lens_name
		zoom_factor = calibration_df[calibration_df['type'] == 'zoom_factor']['misc'].iat[0]
		self.zoom_factor = zoom_factor
		settings_notes = calibration_df[calibration_df['type'] == 'settings_notes']['misc'].iat[0]
		self.settings_notes = settings_notes
		orientation = calibration_df[calibration_df['type'] == 'orientation']['misc'].iat[0]
		calibration_quadrant = calibration_df[calibration_df['type'] == 'quadrant']['misc'].iat[0]
		sensor_dimensions = [int(calibration_df[calibration_df['type'] == 'sensor_dimensions']['px_x'].iat[0]), int(calibration_df[calibration_df['type'] == 'sensor_dimensions']['px_y'].iat[0])]
		self.sensor_dimensions = sensor_dimensions
		max_sensor_index = np.subtract(sensor_dimensions, 1)
		self.max_sensor_index = max_sensor_index
		image_dimensions = [int(calibration_df[calibration_df['type'] == 'image_dimensions']['px_x'].iat[0]), int(calibration_df[calibration_df['type'] == 'image_dimensions']['px_y'].iat[0])]
		self.image_dimensions = image_dimensions
		targets_diameter_degrees = float(calibration_df[calibration_df['type'] == 'targets_diameter']['misc'].iat[0])
		total_sections = int(calibration_df[calibration_df['type'] == 'total_sections']['misc'].iat[0])
		max_image_index = np.subtract(image_dimensions, 1)
		self.max_image_index = max_image_index
		center_px = max_image_index / 2 # usually x.5, non-integer, bc even number of pixels means center is a boundary rather than a pixel
		
		if orientation == 'portrait':
			center_px = center_px[::-1]
			sensor_dimensions = sensor_dimensions[::-1]
			max_sensor_index = max_sensor_index[::-1]
			image_dimensions = image_dimensions[::-1]
			max_image_index = max_image_index[::-1]

		# generate accurate reference pixels from the calibration measurements
		# initialize variables
		section_count = 0
		target_top = 'reset'
		target_right = 'reset'
		target_bottom = 'reset'
		target_left = 'reset'
		grid_align_v_left = 'reset'
		grid_align_v_right = 'reset'
		grid_align_h_top = 'reset'
		grid_align_h_bottom = 'reset'
		section_count = 0
		crosshairs = np.empty((total_sections, 0)).tolist()
		crosshairs[section_count] = center_px
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
			elif 'new_section' in do_what:
				section_count = int((row['misc'].split())[1])
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
				v_align_altaz_reltarget = [row['alt_rel'], row['az_rel']] # is the vertical align target rel to target
			elif do_what == 'grid_align_vertical_right':
				grid_align_v_right = [row['px_x'], row['px_y']]
			elif do_what == 'grid_align_horizontal_top':
				grid_align_h_top = [row['px_x'], row['px_y']]
				h_align_altaz_reltarget = [row['alt_rel'], row['az_rel']] # is the horizontal align target rel to target
			elif do_what == 'grid_align_horizontal_bottom':
				grid_align_h_bottom = [row['px_x'], row['px_y']]
			# if four sides of target have been set, find hit pixel, find altaz adjustment (= position of the target in the image), then re-zero variables
			if 'reset' not in (target_left, target_right, target_top, target_bottom):
				target_pos_px, target_pos_altaz_relaim = target_miss(target_left, target_right, target_top, target_bottom, targets_diameter_degrees, crosshairs[section_count])
				target_top = 'reset'
				target_right = 'reset'
				target_bottom = 'reset'
				target_left = 'reset'
				# print('shooting result, section ', section_count, ': aimed ', crosshairs[section_count], ' target position in image ', target_pos_px, 'actual target to crosshairs altaz ',  target_pos_altaz_relaim, '\n')

			# just finds grid rotation error
			if 'reset' not in (grid_align_v_left, grid_align_v_right, grid_align_h_top, grid_align_h_bottom):
				grid_rotation_error_degreesCCW_v, pixel_delta_v_x = grid_rotation_error(orientation='vertical', grid_align1_px=grid_align_v_left, grid_align2_px=grid_align_v_right, grid_align_altaz_reltarget=v_align_altaz_reltarget, targets_diameter_degrees=targets_diameter_degrees, target_pos_px=target_pos_px)
				grid_rotation_error_degreesCCW_h, pixel_delta_h_y = grid_rotation_error(orientation='horizontal', grid_align1_px=grid_align_h_top, grid_align2_px=grid_align_h_bottom, grid_align_altaz_reltarget=h_align_altaz_reltarget, targets_diameter_degrees=targets_diameter_degrees, target_pos_px=target_pos_px)
				if aim_target_pos_altaz_relcenter[1] != 0: # can only use vertical align point for rotation adjust if on the altitude axis
					grid_rotation_error_degreesCCW_v = 'reset'
				if aim_target_pos_altaz_relcenter[0] != 0: # can only use horizontal align point for rotation adjust if on the azimuth axis
					grid_rotation_error_degreesCCW_h = 'reset'
				if type(grid_rotation_error_degreesCCW_v) in (float, int) and type(grid_rotation_error_degreesCCW_h) in (float, int):
					grid_rotation_error_degreesCCW = (grid_rotation_error_degreesCCW_v + grid_rotation_error_degreesCCW_h) / 2
				elif type(grid_rotation_error_degreesCCW_v) in (float, int):
					grid_rotation_error_degreesCCW = grid_rotation_error_degreesCCW_v
				elif type(grid_rotation_error_degreesCCW_h) in (float, int):
					grid_rotation_error_degreesCCW = grid_rotation_error_degreesCCW_h
				else:
					grid_rotation_error_degreesCCW = 0
				
				# reset variables and move on to grid points
				grid_align_v_left = 'reset'
				grid_align_v_right = 'reset'
				grid_align_h_top = 'reset'
				grid_align_h_bottom = 'reset'
				grid_rotation_error_degreesCCW_v = 'reset'
				grid_rotation_error_degreesCCW_h = 'reset'
			
			if 'grid_point' in do_what:
				theta, r = altaz_to_special_polar(float(row['alt_rel']), float(row['az_rel']))
				grid_rotation_adjust_alt_relgridpoint =  math.tan(grid_rotation_error_degreesCCW*math.pi/180) * r * math.sin(theta*math.pi/180)
				grid_rotation_adjust_az_relgridpoint =  math.tan(grid_rotation_error_degreesCCW*math.pi/180) * r * math.cos(theta*math.pi/180) * -1
				
				if do_what == 'grid_point': # adjust the altaz for grid points
					row_x_px = float(row['px_x'])
					row_y_px = float(row['px_y'])
					row_alt = aim_target_pos_altaz_relcenter[0] + float(row['alt_rel']) + target_pos_altaz_relaim[0] + grid_rotation_adjust_alt_relgridpoint
					row_az = aim_target_pos_altaz_relcenter[1] + float(row['az_rel']) + target_pos_altaz_relaim[1] + grid_rotation_adjust_az_relgridpoint
				elif 'grid_point_align_v' in do_what: # adjust the pixel for align points, not the altaz
					if aim_target_pos_altaz_relcenter[1] == 0: # if on the altitude axis
						row_x_px = center_px[0]
					else:
						row_x_px = float(row['px_x']) + pixel_delta_v_x * (target_pos_altaz_relaim[1] + grid_rotation_adjust_az_relgridpoint)
					row_y_px = float(row['px_y'])
					row_alt = aim_target_pos_altaz_relcenter[0] + float(row['alt_rel'])
					row_az = aim_target_pos_altaz_relcenter[1] + float(row['az_rel'])
					for_section = int(row['misc'].split()[1])
					crosshairs[for_section] = [row_x_px, row_y_px]
				elif 'grid_point_align_h' in do_what:
					row_x_px = float(row['px_x'])
					if aim_target_pos_altaz_relcenter[0] == 0: # if on the azimuth axis
						row_y_px = center_px[1]
					else:
						row_y_px = float(row['px_y']) - pixel_delta_h_y * (target_pos_altaz_relaim[0] + grid_rotation_adjust_alt_relgridpoint)
					row_alt = aim_target_pos_altaz_relcenter[0] + float(row['alt_rel'])
					row_az = aim_target_pos_altaz_relcenter[1] + float(row['az_rel'])
					for_section = int((row['misc'].split())[1])
					crosshairs[for_section] = [row_x_px, row_y_px]

				row_data = [row_x_px, row_y_px, row_alt, row_az]
				reference_pixels.append(row_data)

		reference_pixels = np.array(reference_pixels)
		
		# create the model to predict the rest of the pixels' alt
		reference_pixels[:,0] = np.subtract(reference_pixels[:,0], center_px[0]) # center the x, y coordinates on the center
		reference_pixels[:,1] = np.subtract(reference_pixels[:,1], center_px[1]) # center the x, y coordinates on the center
		# print('reference pixels \n', reference_pixels)
		independent_x_y_px = reference_pixels[:,[0,1]]
		dependent_alt = reference_pixels[:, 2]
		dependent_az = reference_pixels[:, 3]
		alt_model = linear_model.LinearRegression()
		alt_model.fit(independent_x_y_px, dependent_alt)
		az_model = linear_model.LinearRegression()
		az_model.fit(independent_x_y_px, dependent_az)

		# create and fill the appropriate quadrant with x, y coordinate centered at [0, 0]
		pixel_aim = np.empty([image_dimensions[0], image_dimensions[1], 4]) # width rows of height columns means matrix is transpose of image but indexing will then be (x, y) which is what I want bc it matches all image processing including Python's PIL
		pixel_aim[:,:,0], pixel_aim[:,:,1] = np.mgrid[0:image_dimensions[0], 0:image_dimensions[1]]
		pixel_aim[:,:,0] = np.subtract(pixel_aim[:,:,0], center_px[0]) # center the values for the prediction
		pixel_aim[:,:,1] = np.subtract(pixel_aim[:,:,1], center_px[1]) # center the values for the prediction

		# quadrants dimensions, indices, slices
		img_q_dims = np.divide(image_dimensions, 2).astype(int) # assuming even, which they always are as far as I know
		min_Lq_x = 0
		max_Lq_x = img_q_dims[0] - 1
		min_Rq_x = img_q_dims[0]
		max_Rq_x = max_image_index[0]
		min_Tq_y = 0
		max_Tq_y = img_q_dims[1] - 1
		min_Bq_y = img_q_dims[1]
		max_Bq_y = max_image_index[1]
		LT_px_slice = (slice(min_Lq_x, max_Lq_x+1), slice(min_Tq_y, max_Tq_y+1), slice(0, 2))
		LT_alt_slice = (slice(min_Lq_x, max_Lq_x+1), slice(min_Tq_y, max_Tq_y+1), slice(2, 3))
		LT_az_slice = (slice(min_Lq_x, max_Lq_x+1), slice(min_Tq_y, max_Tq_y+1), slice(3, 4))
		RT_px_slice = (slice(min_Rq_x, max_Rq_x+1), slice(min_Tq_y, max_Tq_y+1), slice(0, 2))
		RT_alt_slice = (slice(min_Rq_x, max_Rq_x+1), slice(min_Tq_y, max_Tq_y+1), slice(2, 3))
		RT_az_slice = (slice(min_Rq_x, max_Rq_x+1), slice(min_Tq_y, max_Tq_y+1), slice(3, 4))
		LB_px_slice = (slice(min_Lq_x, max_Lq_x+1), slice(min_Bq_y, max_Bq_y+1), slice(0, 2))
		LB_alt_slice = (slice(min_Lq_x, max_Lq_x+1), slice(min_Bq_y, max_Bq_y+1), slice(2, 3))
		LB_az_slice = (slice(min_Lq_x, max_Lq_x+1), slice(min_Bq_y, max_Bq_y+1), slice(3, 4))
		RB_px_slice = (slice(min_Rq_x, max_Rq_x+1), slice(min_Bq_y, max_Bq_y+1), slice(0, 2))
		RB_alt_slice = (slice(min_Rq_x, max_Rq_x+1), slice(min_Bq_y, max_Bq_y+1), slice(2, 3))
		RB_az_slice = (slice(min_Rq_x, max_Rq_x+1), slice(min_Bq_y, max_Bq_y+1), slice(3, 4))

		if calibration_quadrant == 'LB':
			pixel_aim[LB_alt_slice] = alt_model.predict(pixel_aim[LB_px_slice].reshape(-1,2)).reshape((img_q_dims[0], img_q_dims[1], 1))
			pixel_aim[LB_az_slice] = az_model.predict(pixel_aim[LB_px_slice].reshape(-1,2)).reshape(img_q_dims[0], img_q_dims[1], 1)
			pixel_aim[RB_alt_slice] = np.flipud(pixel_aim[LB_alt_slice])
			pixel_aim[RB_az_slice] = np.flipud(pixel_aim[LB_az_slice] * -1)
			pixel_aim[LT_alt_slice] = np.fliplr(pixel_aim[LB_alt_slice] * -1)
			pixel_aim[LT_az_slice] = np.fliplr(pixel_aim[LB_az_slice])
			pixel_aim[RT_alt_slice] = np.fliplr(pixel_aim[RB_alt_slice] * -1)
			pixel_aim[RT_az_slice] = np.fliplr(pixel_aim[RB_az_slice])

		pixel_aim[:,:,0] = np.add(pixel_aim[:,:,0], center_px[0]) # un-center
		pixel_aim[:,:,1] = np.add(pixel_aim[:,:,1], center_px[1]) # un-center

		self.pixel_aim = pixel_aim
		
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
		"""