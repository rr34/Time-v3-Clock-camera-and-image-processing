import numpy as np
import pandas as pd
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

		if calibration_quadrant == 'LB':
			center_px[0] = (center_px[0] - 0.5).astype(int)
			center_px[1] = (center_px[1] + 0.5).astype(int)

		# generate accurate reference pixels from the calibration measurements
		# initialize variables
		section_count = 0
		target_top = 'reset'
		target_right = 'reset'
		target_bottom = 'reset'
		target_left = 'reset'
		align_v_left = 'reset'
		align_v_right = 'reset'
		align_h_top = 'reset'
		align_h_bottom = 'reset'
		section_count = 0
		crosshairs_log = np.empty((total_sections, 0)).tolist()
		crosshairs_log[section_count] = center_px
		reference_pixels = []
		reference_pixels.append([center_px[0], center_px[1], 0, 0])

		# for loop now row by row because data has to be processed in order
		for row in calibration_df[calibration_df['type'] == 'calibration'].iterrows():
			# prepare the for loop
			row = row[1] # row[0] is just the index. I want the data, don't care about the index
			do_what = row['misc']
			# check for special grid point types and record in appropriate variable
			if do_what == 'end':
				break
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
			elif do_what == 'align_vertical_left':
				align_v_left = [row['px_x'], row['px_y']]
				align_altaz_reltarget = [row['alt_rel'], row['az_rel']] # is the vertical align target rel to target
			elif do_what == 'align_vertical_right':
				align_v_right = [row['px_x'], row['px_y']]
			elif do_what == 'align_horizontal_top':
				align_h_top = [row['px_x'], row['px_y']]
				align_altaz_reltarget = [row['alt_rel'], row['az_rel']] # is the horizontal align target rel to target
			elif do_what == 'align_horizontal_bottom':
				align_h_bottom = [row['px_x'], row['px_y']]
			# if the four sides of target have been set, find hit pixel, find altaz adjustment (= position of the target in the image), then re-zero variables
			if 'reset' not in (target_left, target_right, target_top, target_bottom):
				target_pos_px, target_pos_altaz_relaim = target_miss(target_left, target_right, target_top, target_bottom, targets_diameter_degrees, crosshairs_log[section_count])
				target_top = 'reset'
				target_right = 'reset'
				target_bottom = 'reset'
				target_left = 'reset'
				# print('shooting result, section ', section_count, ': aimed ', crosshairs_log[section_count], ' target position in image ', target_pos_px, 'actual target to crosshairs altaz ',  target_pos_altaz_relaim, '\n')

			# if align, find grid rotation error and pixel delta. BUT if rotation error is invalid, zero it
			if ('reset' not in (align_v_left, align_v_right)) or ('reset' not in (align_h_top, align_h_bottom)):
				if 'horizontal' in do_what:
					align_orientation = 'horizontal'
					align1_px = align_h_top
					align2_px = align_h_bottom
				elif 'vertical' in do_what:
					align_orientation = 'vertical'
					align1_px = align_v_left
					align2_px = align_v_right
				grid_rotation_error_degreesCCW, pixel_delta_at_align = grid_rotation_error(align_orientation=align_orientation, align1_px=align1_px, align2_px=align2_px, align_altaz_reltarget=align_altaz_reltarget, targets_diameter_degrees=targets_diameter_degrees, target_pos_px=target_pos_px)

				if (align_orientation == 'horizontal' and aim_target_pos_altaz_relcenter[0] != 0) or (align_orientation == 'vertical' and aim_target_pos_altaz_relcenter[1] != 0):
					grid_rotation_error_degreesCCW = 0
					# TODO for the case of section zero on both the alt and az axes, average the errors instead of using just one
				
				# reset variables and move on to grid points
				align_v_left = 'reset'
				align_v_right = 'reset'
				align_h_top = 'reset'
				align_h_bottom = 'reset'
			
			# THE ONLY VALUES RECORDED CONTAIN "grid_point", set case-by-case with adjustments, then record at end
			if 'grid_point' in do_what:
				theta, r = altaz_to_special_polar(float(row['alt_rel']), float(row['az_rel']))
				grid_rotation_adjust_alt_relgridpoint =  math.tan(grid_rotation_error_degreesCCW*math.pi/180) * r * math.sin(theta*math.pi/180)
				grid_rotation_adjust_az_relgridpoint =  math.tan(grid_rotation_error_degreesCCW*math.pi/180) * r * math.cos(theta*math.pi/180) * -1
				
				# if normal grid point
				if do_what == 'grid_point': # adjust the altaz for grid points
					row_x_px = float(row['px_x'])
					row_az = aim_target_pos_altaz_relcenter[1] + float(row['az_rel']) + target_pos_altaz_relaim[1] + grid_rotation_adjust_az_relgridpoint
					row_y_px = float(row['px_y'])
					row_alt = aim_target_pos_altaz_relcenter[0] + float(row['alt_rel']) + target_pos_altaz_relaim[0] + grid_rotation_adjust_alt_relgridpoint
				# if a vertical align grid point, set values, then set pixel coordinate as crosshairs for next section target
				elif 'grid_point_crosshairs' in do_what: # adjust the pixel for align points, not the altaz
					row_x_px = float(row['px_x']) + pixel_delta_at_align * (target_pos_altaz_relaim[1] + grid_rotation_adjust_az_relgridpoint)
					row_az = aim_target_pos_altaz_relcenter[1] + float(row['az_rel'])
					row_y_px = float(row['px_y'])
					row_alt = aim_target_pos_altaz_relcenter[0] + float(row['alt_rel'])
					for_section = int(row['misc'].split()[1])
					crosshairs_log[for_section] = [row_x_px, row_y_px]
					# TODO: for case where crosshairs can come from either of two previous sections, average the location.
				# if a vertical align grid point, set values, then set pixel coordinate as crosshairs for next section target
				elif 'grid_point_crosshairs' in do_what:
					row_x_px = float(row['px_x'])
					row_az = aim_target_pos_altaz_relcenter[1] + float(row['az_rel'])
					if aim_target_pos_altaz_relcenter[0] == 0:
						row_y_px = center_px[1]
						row_y_px = float(row['px_y']) - pixel_delta_at_align * (target_pos_altaz_relaim[0] + grid_rotation_adjust_alt_relgridpoint)
					row_alt = aim_target_pos_altaz_relcenter[0] + float(row['alt_rel'])
					for_section = int((row['misc'].split())[1])
					crosshairs_log[for_section] = [row_x_px, row_y_px]

				"""
				# if grid point is on azimuth axis, force x to center and az to zero
				if aim_target_pos_altaz_relcenter[1] + float(row['az_rel']) == 0:
					row_x_px = center_px[0]
					row_az = 0
				# if grid point is on altitude axis, force y to center and alt to zero
				if aim_target_pos_altaz_relcenter[0] + float(row['alt_rel']) == 0:
					row_y_px = center_px[1]
					row_alt = 0
				"""

				row_data = [row_x_px, row_y_px, row_alt, row_az]
				reference_pixels.append(row_data)

		reference_pixels = np.array(reference_pixels)
		# print('reference pixels \n', reference_pixels)

		# create the model to predict the rest of the pixels' alt and az
		from sklearn.preprocessing import StandardScaler, PolynomialFeatures
		from sklearn.linear_model import LinearRegression
		# copy the calibration (x, y) pixel coordinate data as independent, then center, then scale for model creation
		independent_x_y_px = np.empty(reference_pixels[:,[0,1]].shape)
		independent_x_y_px = reference_pixels[:,[0,1]]
		independent_x_y_px[:,0] = np.subtract(independent_x_y_px[:,0], center_px[0])
		independent_x_y_px[:,1] = np.subtract(independent_x_y_px[:,1], center_px[1])
		scale = StandardScaler()
		# independent_x_y_px = scale.fit_transform(independent_x_y_px)
		# now the alt and az as dependent data - notice pointing to actual reference_pixels memory location, not copying the data
		dependent_alt = reference_pixels[:, 2]
		dependent_az = reference_pixels[:, 3]
		# generate a polynomial fit model using the calibration data
		poly = PolynomialFeatures(degree=3, include_bias=False) #generate a generic PolynomialFeatures class
		independent_poly = poly.fit_transform(independent_x_y_px)
		alt_model = LinearRegression()
		alt_model.fit(independent_poly, dependent_alt)
		az_model = LinearRegression()
		az_model.fit(independent_poly, dependent_az)

		# create empty full-size matrix, fill (x, y) pixel coordinates, then put (0, 0) at center, then scale for prediction with model
		pixel_aim = np.empty([image_dimensions[0], image_dimensions[1], 4]) # width rows of height columns means matrix is transpose of image but index is (x, y) which is what I want bc it matches all image processing including Python's PIL
		pixel_aim[:,:,0], pixel_aim[:,:,1] = np.mgrid[0:image_dimensions[0], 0:image_dimensions[1]] # 0 is x values, 1 is y values

		# whole matrix slices
		px_slice = (slice(0, max_image_index[0]+1), slice(0, max_image_index[1]+1), slice(0, 2))
		alt_slice = (slice(0, max_image_index[0]+1), slice(0, max_image_index[1]+1), slice(2, 3))
		az_slice = (slice(0, max_image_index[0]+1), slice(0, max_image_index[1]+1), slice(3, 4))

		# quadrants dimensions, indices, slices used for filling the matrix quadrant-by-quadrant
		img_q_dims = np.divide(image_dimensions, 2).astype(int) # is integer assuming even dims, which they always are as far as I know
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

		# other slices
		center_square_slice = (slice(max_Lq_x-3, min_Rq_x+3), slice(max_Tq_y-3, min_Bq_y+3), slice(0, 4))
		center_h_slice = (slice(max_Lq_x-1, min_Rq_x+2), slice(max_Tq_y-10, min_Bq_y+10), slice(0, 4))

		if calibration_quadrant == 'LB':
			x_y_px = np.empty(pixel_aim[LB_px_slice].shape)
			x_y_px[:,:] = pixel_aim[LB_px_slice]
			x_y_px[:,:,0] = np.subtract(x_y_px[:,:,0], center_px[0])
			x_y_px[:,:,1] = np.subtract(x_y_px[:,:,1], center_px[1])
			x_y_px = x_y_px.reshape(-1,2)
			# x_y_px = scale.fit_transform(x_y_px)
			x_y_px_poly = poly.fit_transform(x_y_px)
			pixel_aim[LB_alt_slice] = alt_model.predict(x_y_px_poly).reshape((img_q_dims[0], img_q_dims[1], 1))
			pixel_aim[LB_az_slice] = az_model.predict(x_y_px_poly).reshape((img_q_dims[0], img_q_dims[1], 1))
			print(center_px, pixel_aim[center_h_slice])
			pixel_aim[RB_alt_slice] = np.flipud(pixel_aim[LB_alt_slice])
			pixel_aim[RB_az_slice] = np.flipud(pixel_aim[LB_az_slice] * -1)
			pixel_aim[LT_alt_slice] = np.fliplr(pixel_aim[LB_alt_slice] * -1)
			pixel_aim[LT_az_slice] = np.fliplr(pixel_aim[LB_az_slice])
			pixel_aim[RT_alt_slice] = np.fliplr(pixel_aim[RB_alt_slice] * -1)
			pixel_aim[RT_az_slice] = np.fliplr(pixel_aim[RB_az_slice])

		if False:
			x_y_px = np.empty(pixel_aim[px_slice].shape)
			x_y_px[:,:] = pixel_aim[px_slice]
			x_y_px = x_y_px.reshape(-1,2)
			x_y_px[:,0] = np.subtract(x_y_px[:,0], center_px[0])
			x_y_px[:,1] = np.subtract(x_y_px[:,1], center_px[1])
			# x_y_px = scale.fit_transform(x_y_px)
			x_y_px_poly = poly.fit_transform(x_y_px)
			pixel_aim[alt_slice] = alt_model.predict(x_y_px_poly).reshape((image_dimensions[0], image_dimensions[1], 1))
			pixel_aim[az_slice] = az_model.predict(x_y_px_poly).reshape((image_dimensions[0], image_dimensions[1], 1))
		
		self.pixel_aim = pixel_aim
		self.reference_pixels = reference_pixels