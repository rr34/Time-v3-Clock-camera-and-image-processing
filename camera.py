import math
import numpy as np
import pandas as pd
from PIL import Image, PngImagePlugin
import pickle


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
	TODO: make a visual version of reference pixel data entry to avoid errors. It is really a mind-bend!
	"""

	def __init__(self, calibration_file):
		"""
		Parameters.
		----------
		camera_name : the name of the camera together with some lens and settings notes for quick
			identification as each 
			
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
		# most parameters (all actually?) are defined with a calibration CSV file
		cal_df = pd.read_csv(calibration_file)

		# calibration data includes most of requirements to initialize the object:
		self.camera_name = cal_df[cal_df['type'] == 'camera_name']['misc'].iat[0]
		self.lens_name = cal_df[cal_df['type'] == 'lens_name']['misc'].iat[0]
		self.zoom_factor = cal_df[cal_df['type'] == 'zoom_factor']['misc'].iat[0]
		self.settings_notes = cal_df[cal_df['type'] == 'settings_notes']['misc'].iat[0]
		self.cam_image_dimensions = [int(cal_df[cal_df['type'] == 'cam_image_dimensions']['x_rec'].iat[0]), int(cal_df[cal_df['type'] == 'cam_image_dimensions']['y_rec'].iat[0])]
		self.max_image_index = np.subtract(self.cam_image_dimensions, 1)
		self.pixel_map_type = cal_df[cal_df['type'] == 'pixel_map_type']['misc'].iat[0]
		self.center_px = self.max_image_index / 2 # usually x.5, non-integer, bc even number of pixels means center is a boundary rather than a pixel
		calibration_orientation = cal_df[cal_df['type'] == 'orientation']['misc'].iat[0]
		calibration_quadrant = cal_df[cal_df['type'] == 'quadrant']['misc'].iat[0]
		self.calibration_distance_cm = float(cal_df[cal_df['type'] == 'distance']['misc'].iat[0])
		align_targets_radius_cm = float(cal_df[cal_df['type'] == 'align_targets_radius']['misc'].iat[0])

		if calibration_orientation == 'portrait':
			self.center_px = self.center_px[::-1]
			self.cam_image_dimensions = self.cam_image_dimensions[::-1]
			self.max_image_index = self.max_image_index[::-1]

		for row in cal_df[cal_df['type'] == 'calculate_and_record'].iterrows():
			row_xycm = [float(row[1]['x_cm']), float(row[1]['y_cm'])]
			target_pos_px = cal_df[cal_df['misc']=='target_center'][['x_rec', 'y_rec']].values[0]

			if row[1]['misc'] == 'target_pos_xyangs_relcenter':
				x_small_cm = cal_df[cal_df['misc']=='x_small'][['x_cm', 'y_cm']].values[0]
				x_small_px = cal_df[cal_df['misc']=='x_small'][['x_rec', 'y_rec']].values[0]
				y_small_cm = cal_df[cal_df['misc']=='y_small'][['x_cm', 'y_cm']].values[0]
				y_small_px = cal_df[cal_df['misc']=='y_small'][['x_rec', 'y_rec']].values[0]

				target_pos_xycm_relaim = _target_miss(self, target_pos_px, x_small_cm, x_small_px, y_small_cm, y_small_px)
				cal_df.loc[row[0], 'out1'] = target_pos_xycm_relaim[0]
				cal_df.loc[row[0], 'out2'] = target_pos_xycm_relaim[1]

			elif row[1]['misc'] == 'rotation_error_side_degCCW':
				align_orientation = 'vertical'
				align1_px = cal_df[cal_df['misc']=='align_v_center'][['x_rec', 'y_rec']].values[0]
				align2_px = cal_df[cal_df['misc']=='align_v_left'][['x_rec', 'y_rec']].values[0]
				if not math.isnan(align1_px[0]) and not math.isnan(align1_px[1]) and not math.isnan(align2_px[0]) and not math.isnan(align2_px[1]): # IF NOT BLANK
					grid_rotation_error_degreesCCW = _grid_rotation_error(self, row_xycm, align_orientation, align1_px, align2_px, align_targets_radius_cm, target_pos_px)
					cal_df.loc[row[0], 'out1'] = grid_rotation_error_degreesCCW
			elif row[1]['misc'] == 'rotation_error_top_degCCW':
				align_orientation = 'horizontal'
				align1_px = cal_df[cal_df['misc']=='align_h_center'][['x_rec', 'y_rec']].values[0]
				align2_px = cal_df[cal_df['misc']=='align_h_down'][['x_rec', 'y_rec']].values[0]
				if not math.isnan(align1_px[0]) and not math.isnan(align1_px[1]) and not math.isnan(align2_px[0]) and not math.isnan(align2_px[1]): # IF NOT BLANK
					grid_rotation_error_degreesCCW = _grid_rotation_error(self, row_xycm, align_orientation, align1_px, align2_px, align_targets_radius_cm, target_pos_px)
					cal_df.loc[row[0], 'out1'] = grid_rotation_error_degreesCCW
			elif row[1]['misc'] == 'average_rotation_error_degCCW':
				align_v_degCCW = float(cal_df[cal_df['misc']=='rotation_error_side_degCCW'][['out1']].values[0])
				align_h_degCCW = float(cal_df[cal_df['misc']=='rotation_error_top_degCCW'][['out1']].values[0])
				average_rotation_error_degCCW = (align_v_degCCW + align_h_degCCW)/2
				cal_df.loc[row[0], 'out1'] = average_rotation_error_degCCW
				
		# second for loop: each reference point represented by a "ref_point" row
		# ref point case 1: record the center as a reference point bc defining as original xyangs = [0,0]
		# ref point case 2: like crosshairs case 2, on xcm axis (vertical)
		# ref point case 3: like crosshairs case 3. on ycm axis (horizontal)
		# ref point case 4: normal ref point
		for row in cal_df[cal_df['type'] == 'ref_point'].iterrows():
			px = row[1][['x_rec', 'y_rec']].values
			row_xycm = [float(row[1]['x_cm']), float(row[1]['y_cm'])]

			if not math.isnan(px[0]) and not math.isnan(px[1]): # IF NOT BLANK

				# record the ref pixels for the 4 cases
				if row_xycm[0]==0 and row_xycm[1]==0: # origin
					cal_df.loc[row[0], 'x_px'] = self.center_px[0]
					cal_df.loc[row[0], 'y_px'] = self.center_px[1]
					cal_df.loc[row[0], 'xang'] = 0.0
					cal_df.loc[row[0], 'yang'] = 0.0
				elif row_xycm[0]==0 and row_xycm[1]!=0: # on the vertical axis just take the px distance and put on the line with nominal cm value
					px_dist = math.sqrt((px[0] - target_pos_px[0])**2 + (px[1] - target_pos_px[1])**2)
					cal_df.loc[row[0], 'x_px'] = self.center_px[0]
					cal_df.loc[row[0], 'y_px'] = self.center_px[1] + px_dist
					yang = math.atan(row_xycm[1]/self.calibration_distance_cm) * 180/math.pi # (-) as appropriate bc (-) ycm and (+) distance
					cal_df.loc[row[0], 'xang'] = 0.0
					cal_df.loc[row[0], 'yang'] = yang
				elif row_xycm[0]!=0 and row_xycm[1]==0: # on the horizontal axis just take the px distance and put on the line with nominal cm value
					px_dist = math.sqrt((px[0] - target_pos_px[0])**2 + (px[1] - target_pos_px[1])**2)
					cal_df.loc[row[0], 'x_px'] = self.center_px[0] - px_dist
					cal_df.loc[row[0], 'y_px'] = self.center_px[1]
					xang = math.atan(row_xycm[0]/self.calibration_distance_cm) * 180/math.pi # (-) as appropriate bc (-) xcm and (+) distance
					cal_df.loc[row[0], 'xang'] = xang
					cal_df.loc[row[0], 'yang'] = 0.0
				else: # anywhere else
					row_polar = _xycm_to_polar(row_xycm)
					rotation_cmatpt_hypotenuse = math.tan(average_rotation_error_degCCW*math.pi/180) * row_polar[0]
					rotation_x_cmatpt = rotation_cmatpt_hypotenuse * -1*math.sin(row_polar[1]*math.pi/180) # cm measurement directions are (+) right and up
					rotation_y_cmatpt = rotation_cmatpt_hypotenuse * 1*math.cos(row_polar[1]*math.pi/180)

					actual_x_cm_rel_center = row_xycm[0] + target_pos_xycm_relaim[0] + rotation_x_cmatpt # eg in LB quadrant, if target pos (+), gets added to (-) xycm and distance is less
					actual_y_cm_rel_center = row_xycm[1] + target_pos_xycm_relaim[1] + rotation_y_cmatpt
					
					# trigonometry: there is very little for the calibration bc it isolates the camera with other variables being zero.
					r1 = self.calibration_distance_cm
					r2 = r1 # because the cal board is flat, not a sphere
					yang = math.atan(actual_y_cm_rel_center/r2) * 180/math.pi # mostly (-) because y_cm is (-). the cal board lines would project a small circle on a sphere, so correct
					xang = math.atan(actual_x_cm_rel_center/r1) * 180/math.pi  # is the same as az_rel from diagram bc camera is level and perpendicular to the board
					xyangs = [xang,yang]

					cal_df.loc[row[0], 'x_px'] = px[0]
					cal_df.loc[row[0], 'y_px'] = px[1]
					cal_df.loc[row[0], 'xang'] = xyangs[0]
					cal_df.loc[row[0], 'yang'] = xyangs[1]

		cal_df.to_csv('slay cal output.csv')
		ref_df_filter = pd.notnull(cal_df['x_px'])
		ref_df = pd.DataFrame(cal_df[ref_df_filter][['x_px', 'y_px', 'xang', 'yang']])

		# standardize the reference pixels:
		# center, convert to positive values (RB quadrant in PhotoShop coordinate system), then orient landscape
		ref_df['x_px'] = ref_df['x_px'] - self.center_px[0]
		ref_df['y_px'] = ref_df['y_px'] - self.center_px[1]
		if calibration_quadrant == 'LB':
			ref_df['x_px'] = ref_df['x_px'] * -1
			ref_df['y_px'] = ref_df['y_px'] * -1 * -1 # twice because y PSpx is (-) down
			ref_df['xang'] = ref_df['xang'] * -1
			ref_df['yang'] = ref_df['yang'] * -1
		if calibration_orientation == 'portrait':
			self.center_px = self.center_px[::-1]
			self.cam_image_dimensions = self.cam_image_dimensions[::-1]
			self.max_image_index = self.max_image_index[::-1]
			ref_df['x_px'], ref_df['y_px'] = ref_df['y_px'], ref_df['x_px']
			ref_df['yang'], ref_df['xang'] = ref_df['xang'], ref_df['yang']

		# create the px to xyangs models
		from sklearn.preprocessing import PolynomialFeatures
		from sklearn.linear_model import LinearRegression

		poly1 = PolynomialFeatures(degree=3, include_bias=False)
		poly2 = PolynomialFeatures(degree=3, include_bias=False)

		ref_df.to_csv('slay ref df.csv')

		independent_poly_px = pd.DataFrame(data=poly1.fit_transform(ref_df[['x_px', 'y_px']]), columns=poly1.get_feature_names_out(ref_df[['x_px', 'y_px']].columns))
		xyangs_model = LinearRegression(fit_intercept=False)
		xyangs_model.fit(independent_poly_px, ref_df[['xang', 'yang']])

		independent_poly_xyangs = pd.DataFrame(data=poly2.fit_transform(ref_df[['xang', 'yang']]), columns=poly2.get_feature_names_out(ref_df[['xang', 'yang']].columns))
		px_model = LinearRegression(fit_intercept=False)
		px_model.fit(independent_poly_xyangs, ref_df[['x_px', 'y_px']])

		self.ref_df = ref_df
		self.poly1 = poly1
		self.poly2 = poly2
		self.xyangs_model = xyangs_model
		self.px_model = px_model

		px_LT = [0-self.center_px[0], self.center_px[1]]
		px_RT = [self.center_px[0], self.center_px[1]]
		px_LB = [0-self.center_px[0], 0-self.center_px[1]]
		px_RB = [self.center_px[0], 0-self.center_px[1]]
		self.px_corners = np.concatenate((px_LT, px_RT, px_LB, px_RB)).reshape(-1,2)
		xyangs_LT, xyangs_RT, xyangs_LB, xyangs_RB = self.px_xyangs_models_convert(input=self.px_corners, direction='px_to_xyangs')
		self.xyangs_corners = np.concatenate((xyangs_LT, xyangs_RT, xyangs_LB, xyangs_RB)).reshape(-1,2)
		px_top = [0, self.center_px[1]]
		px_right = [self.center_px[0], 0]
		px_bottom = [0, 0-self.center_px[1]]
		px_left = [0-self.center_px[0], 0]
		self.px_edges = np.concatenate((px_top, px_right, px_bottom, px_left)).reshape(-1,2)
		xyangs_top, xyangs_right, xyangs_bottom, xyangs_left = self.px_xyangs_models_convert(input=self.px_edges, direction='px_to_xyangs')
		self.xyangs_edges = np.concatenate((xyangs_top, xyangs_right, xyangs_bottom, xyangs_left)).reshape(-1,2)


	def represent_camera(self):
		print('\nCamera Name: ', self.camera_name)
		print('Lens: ', self.lens_name)
		print('Zoom: ', self.zoom_factor)
		print('Settings Notes: ', self.settings_notes)
		print('Image Dimensions: ', self.cam_image_dimensions)
		print('Pixel Corners LT, RT, LB, RB:\n', self.px_corners)
		print('x,y Angle Corners LT, RT, LB, RB:\n', self.xyangs_corners)
		print('Pixel Edges Top, Right, Bottom, Left:\n', self.px_edges)
		print('x,y Angle Edges Top, Right, Bottom, Left:\n', self.xyangs_edges)
		print('Pixels per degree horizontal: ', self.cam_image_dimensions[0] / (2*self.xyangs_edges[1,0]))
		print('Pixels per degree vertical: ', self.cam_image_dimensions[1] / (2*self.xyangs_edges[0,1]))

	# generate awim data in form of a single dictionary for embedding in any image file
	def awim_metadata_generate(self, current_image, date_gregorian_ns_time_utc, earth_latlng, center_ref, azalt_ref, img_orientation, img_tilt):
		img_dimensions = current_image.size
		if img_orientation == 'portrait':
			self.cam_image_dimensions = [self.cam_image_dimensions[1], self.cam_image_dimensions[0]]
		img_aspect_ratio = img_dimensions[0] / img_dimensions[1]
		cam_aspect_ratio = self.cam_image_dimensions[0] / self.cam_image_dimensions[1]
		if img_aspect_ratio != cam_aspect_ratio:
			print('error: image aspect ratio does not match camera aspect ratio, but it should')
		else:
			img_resize_factor = img_dimensions[0] / self.cam_image_dimensions[0]
		max_img_index = np.subtract(img_dimensions, 1)
		img_center = np.divide(max_img_index, 2)
		if center_ref == 'center':
			center_ref = img_center

		if img_orientation == 'landscape':
			xyangs_model_coefficients = pd.DataFrame(self.xyangs_model.coef_, columns=self.xyangs_model.feature_names_in_, index=['xang_predict', 'yang_predict'])
			px_model_coefficients = pd.DataFrame(self.px_model.coef_, columns=self.px_model.feature_names_in_, index=['x_px_predict', 'y_px_predict'])
		elif img_orientation == 'portrait': # TODO: make this the transpose of landscape / swap x and y
			xyangs_model_coefficients = pd.DataFrame(self.xyangs_model.coef_, columns=self.xyangs_model.feature_names_in_, index=['xang_predict', 'yang_predict'])
			px_model_coefficients = pd.DataFrame(self.px_model.coef_, columns=self.px_model.feature_names_in_, index=['x_px_predict', 'y_px_predict'])
		xyangs_model_coefficients /= img_resize_factor
		px_model_coefficients *= img_resize_factor

		px_LT = [0-img_center[0], img_center[1]]
		px_top = [0, img_center[1]]
		px_RT = [img_center[0], img_center[1]]
		px_left = [0-img_center[0], 0]
		px_center = [0, 0]
		px_right = [img_center[0], 0]
		px_LB = [0-img_center[0], 0-img_center[1]]
		px_bottom = [0, 0-img_center[1]]
		px_RB = [img_center[0], 0-img_center[1]]
		img_px_borders = np.concatenate((px_LT, px_top, px_RT, px_left, px_center, px_right, px_LB, px_bottom, px_RB)).reshape(-1,2)
		img_xyangs_borders = self.px_xyangs_models_convert(input=np.divide(img_px_borders, img_resize_factor), direction='px_to_xyangs')

		earth_latlng_string = ', '.join(str(i) for i in earth_latlng)
		img_dimensions_string = ', '.join(str(i) for i in img_dimensions)
		center_ref_string = ', '.join(str(i) for i in center_ref)
		azalt_ref_string = ', '.join(str(i) for i in azalt_ref)
		xyangs_model_coefficients_csv = xyangs_model_coefficients.to_csv()
		px_model_coefficients_csv = px_model_coefficients.to_csv()
		px_borders_string = ', '.join(str(i) for i in img_px_borders)
		xyangs_borders_string = ', '.join(str(i) for i in img_xyangs_borders)

		awim_dictionary = {'Location': earth_latlng_string, 'Capture Moment': date_gregorian_ns_time_utc.isoformat(timespec='seconds'), 'Dimensions': img_dimensions_string, 'Center Pixel': center_ref_string, 'Center AzAlt': azalt_ref_string, 'Pixel Models': px_model_coefficients_csv, 'Pixel Map Type': self.pixel_map_type, 'x,y Angle Models': xyangs_model_coefficients_csv, 'Pixel Borders': px_borders_string, 'x,y Angle Borders': xyangs_borders_string}

		return awim_dictionary


	def px_xyangs_models_convert(self, input, direction):
		if isinstance(input, list): # models require numpy arrays
			input = np.array(input, dtype=float).reshape(-1,2)

		full_shape = input.shape # save the shape for reshape later
		output = np.zeros(full_shape)
		output_sign = np.where(input < 0, -1, 1) # models are positive values only. Save sign.
		input = np.abs(input)
		input = input.reshape(-1,2)
		if direction == 'px_to_xyangs':
			input_poly = self.poly1.fit_transform(input)
			output = self.xyangs_model.predict(input_poly).reshape(full_shape)
		elif direction == 'xyangs_to_px':
			input_poly = self.poly2.fit_transform(input)
			output = self.px_model.predict(input_poly).reshape(full_shape)
		output = np.multiply(output, output_sign)

		return output

# functions below here
# theta is CCW from the positive x axis. ar is cm from center
def _xycm_to_polar(xycm):
	ar = math.sqrt(xycm[0]**2 + xycm[1]**2)

	if xycm[0] == 0 and xycm[1] == 0:
		theta_rad = 0
	elif xycm[0] >= 0 and xycm[1] == 0:
		theta_rad = 0
	elif xycm[0] == 0 and xycm[1] > 0:
		theta_rad = math.pi/2
	elif xycm[0] < 0 and xycm[1] == 0:
		theta_rad = math.pi
	elif xycm[0] == 0 and xycm[1] < 0:
		theta_rad = 3/2*math.pi
	elif xycm[0] > 0 and xycm[1] != 0:
		theta_rad = (math.atan(xycm[1] / xycm[0])) % (2*math.pi)
	elif xycm[0] < 0 and xycm[1] != 0:
		theta_rad = math.atan(xycm[1] / xycm[0]) + math.pi

	theta = theta_rad * 180/math.pi

	return  [ar, theta]

# cool matrix math from StackOverflow:
def _intersect_two_lines(line_1_1, line_1_2, line_2_1, line_2_2):
	right_rel = np.subtract(line_1_2,line_1_1)
	bottom_rel = np.subtract(line_2_2,line_2_1)
	LR_scalar = np.cross(np.subtract(line_2_1,line_1_1), bottom_rel) / np.cross(right_rel, bottom_rel)
	TB_scalar = np.cross(np.subtract(line_1_1,line_2_1), right_rel) / np.cross(bottom_rel, right_rel)
	if np.cross(right_rel, bottom_rel) == 0 and np.cross(np.subtract(line_2_1,line_1_1), right_rel) == 0:
		return 'error, collinear lines given'
	if np.cross(right_rel, bottom_rel) == 0 and np.cross(np.subtract(line_2_1,line_1_1), right_rel) != 0:
		return 'error, parallel non-intersecting lines given'
	if np.cross(right_rel, bottom_rel) != 0 and LR_scalar >= 0 and LR_scalar <= 1 and TB_scalar >= 0 and TB_scalar <= 1:
		intersect = np.add(line_1_1, LR_scalar*right_rel)
		intersect2 = np.add(line_2_1, TB_scalar*bottom_rel) # would be the exact same as the other calculation
		if all(intersect) != all(intersect2):
			return 'error, something went wrong, dont know what, look here'
	return intersect

# calculate the miss angle as relative xang,yang
def _target_miss(self, target_pos_px, x_small_cm, x_small_px, y_small_cm, y_small_px):
	target_x_radius_deg = abs(math.atan(x_small_cm[0] / self.calibration_distance_cm)) * 180/math.pi
	target_y_radius_deg = abs(math.atan(y_small_cm[1] / self.calibration_distance_cm)) * 180/math.pi
	target_x_radius_px = math.sqrt((target_pos_px[0]-x_small_px[0])**2 + (target_pos_px[1]-x_small_px[1])**2)
	target_y_radius_px = math.sqrt((target_pos_px[0]-y_small_px[0])**2 + (target_pos_px[1]-y_small_px[1])**2)
	pixel_delta_x_cm = abs(x_small_cm[0] / target_x_radius_px) # cm per pixel horizontal
	pixel_delta_y_cm = abs(y_small_cm[1] / target_y_radius_px) # cm per pixel vertical
	pixel_delta_x_deg = target_x_radius_deg / target_x_radius_px # deg per pixel horizontal
	pixel_delta_y_deg = target_y_radius_deg / target_y_radius_px # deg per pixel vertical
	target_pos_xcm_rel = (target_pos_px[0] - self.center_px[0]) * pixel_delta_x_cm # camera aim error xang
	target_pos_ycm_rel = -1 * (target_pos_px[1] - self.center_px[1]) * pixel_delta_y_cm # camera aim error yang, note (-)y_px is (+)yang!

	target_pos_xycm_relaim = [target_pos_xcm_rel, target_pos_ycm_rel]
	return target_pos_xycm_relaim

# from two edge pixel coordinates of known-angular-size grid alignment point, angular distance of the alignment point from grid target, and the target position in the image, calculate the angular rotation error of the entire grid,
# CCW grid rotation in photo is positive.
def _grid_rotation_error(self, row_xycm, align_orientation, align1_px, align2_px, align_targets_radius_cm, target_pos_px):
	# find error rotation angle using vertical grid align target
	if align_orientation == 'horizontal':
		xy_index = 1 # align with the y pixel coordinate
		xycm_index = 0 # use the x_cm for distance from target
	elif align_orientation == 'vertical':
		xy_index = 0 # align with the x pixel coordinate
		xycm_index = 1 # use the y_cm for distance from target
	else:
		print('must specify align_orientation')

	align_radius_px = np.sqrt((align2_px[0] - align1_px[0])**2 + (align2_px[1] - align1_px[1])**2)
	pixel_delta_cmpx = align_targets_radius_cm / align_radius_px
	miss_cm = (align1_px[xy_index] - target_pos_px[xy_index]) * pixel_delta_cmpx # grid align pixel not in same pixel line with hit target pixel. should be bc on axis. align1_px right or down from target is CCW and (+)
	grid_rotation_error_degreesCCW = math.atan(miss_cm/row_xycm[xycm_index]) * 180/math.pi

	return grid_rotation_error_degreesCCW

# simply standardize coordinate type:
# single coordinate is a list. more than one is a 2-dim Numpy array of two columns
def _coord_standard(obj):
	some_data = False
	number_of_rows = 0

	if isinstance(obj, pd.DataFrame):
		if obj.empty:
			description = 'Completely empty DataFrame'
		elif len(obj.index) == 1 and obj.shape[1] == 2:
			if all(pd.notnull(obj).values[0]):
				some_data = True
				description = 'DataFrame containing one coordinate pair'
				obj_shape = (1,2)
				obj = [obj.values[0,0], obj.values[0,1]]
		elif len(obj.index) == 2 and obj.shape[1] == 2:
			if all(pd.notnull(obj).values[0]):
				some_data = True
				description = 'DataFrame containing two coordinate pairs'
				obj_shape = (2,2)
				obj = np.array([obj.values[0], obj.values[1]])
		elif obj.shape[1] == 2:
			if all(pd.notnull(obj).values[0]):
				some_data = True
				description = 'DataFrame containing multiple coordinate pairs'
				obj_shape = ('more than two',2)
				obj = obj.values
				obj = obj.reshape(-1,2)
		else:
			print('Some other condition?')
	elif isinstance(obj, np.ndarray):
		if any(np.isnan(obj)):
			description = 'Numpy array with some non-values'
			# TODO (?) handle values that are present
		elif obj.ndim == 1 and obj.size == 2:
			obj = [obj[0], obj[1]]
		elif obj.ndim == 2 and obj.shape[0] == 1 and obj.shape[1] == 2 and all(~np.isnan(obj)):
			some_data = True
			description = 'Numpy array of one coordinate pair'
			obj_shape = (1,2)
			obj = [obj[0,0], obj[0,1]]
		elif obj.ndim == 2 and all(~np.isnan(obj)):
			some_data = True
			description = 'Numpy array of multiple coordinate pairs'
			obj = obj.reshape(-1,2)
		elif obj.ndim == 3:
			print('handle this?')
		else:
			print('some other condition?')
	
	return some_data, obj