import math
import numpy as np
import pandas as pd
import os
import PIL
import json
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import awimlib, metadata_tools


# theta is CCW from the positive x axis. r is cm from center
def _xycm_to_polar(xycm):
	r = math.sqrt(xycm[0]**2 + xycm[1]**2)

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

	return  [r, theta]


# works 11 Oct 2022 # calculate the miss angle as xang,yang
def _target_miss(calibration_distance_cm, center_px, target_pos_px, x_small_cm, x_small_px, y_small_cm, y_small_px):
	target_x_radius_deg = abs(math.atan(x_small_cm[0] / calibration_distance_cm)) * 180/math.pi
	target_y_radius_deg = abs(math.atan(y_small_cm[1] / calibration_distance_cm)) * 180/math.pi
	target_x_radius_px = math.sqrt((target_pos_px[0]-x_small_px[0])**2 + (target_pos_px[1]-x_small_px[1])**2)
	target_y_radius_px = math.sqrt((target_pos_px[0]-y_small_px[0])**2 + (target_pos_px[1]-y_small_px[1])**2)
	pixel_delta_x_cm = abs(x_small_cm[0] / target_x_radius_px) # cm per pixel horizontal
	pixel_delta_y_cm = abs(y_small_cm[1] / target_y_radius_px) # cm per pixel vertical
	pixel_delta_x_deg = target_x_radius_deg / target_x_radius_px # deg per pixel horizontal
	pixel_delta_y_deg = target_y_radius_deg / target_y_radius_px # deg per pixel vertical
	target_pos_xcm_rel = (target_pos_px[0] - center_px[0]) * pixel_delta_x_cm # camera aim error xang
	target_pos_ycm_rel = -1 * (target_pos_px[1] - center_px[1]) * pixel_delta_y_cm # camera aim error yang, note (-)y_px is (+)yang!

	target_pos_xycm_relaim = [target_pos_xcm_rel, target_pos_ycm_rel]
	return target_pos_xycm_relaim


def _grid_rotation_error(row_xycm, align_orientation, align1_px, align2_px, align_targets_radius_cm, target_pos_px):
	if align_orientation == 'horizontal':
		xy_index = 1 # align with the y pixel coordinate for the horizontal axis
		xycm_index = 0 # use the x_cm for distance from target
	elif align_orientation == 'vertical':
		xy_index = 0 # align with the x pixel coordinate for the vertical axis
		xycm_index = 1 # use the y_cm for distance from target
	else:
		print('must specify align_orientation')

	align_radius_px = np.sqrt((align2_px[0] - align1_px[0])**2 + (align2_px[1] - align1_px[1])**2)
	pixel_delta_cmpx = align_targets_radius_cm / align_radius_px
	miss_cm = (align1_px[xy_index] - target_pos_px[xy_index]) * pixel_delta_cmpx # grid align pixel not in same pixel line with hit target pixel. should be bc on axis. align1_px right or down from target is CCW and (+)
	grid_rotation_error_degreesCCW = math.atan(miss_cm/row_xycm[xycm_index]) * 180/math.pi

	return grid_rotation_error_degreesCCW


# works 11 Oct 2022
def generate_camera_AWIM_from_calibration(calibration_image_path, calibration_file_path):

	calimg_exif_readable = metadata_tools.get_metadata(calibration_image_path)
	cal_df = pd.read_excel(calibration_file_path)

	camera_name = cal_df[cal_df['type'] == 'camera_name']['misc'].iat[0]
	lens_name = cal_df[cal_df['type'] == 'lens_name']['misc'].iat[0]
	focal_length = cal_df[cal_df['type'] == 'focal_length']['misc'].iat[0]
	AWIM_cal_type = cal_df[cal_df['type'] == 'AWIM_calibration_type']['misc'].iat[0]
	pixel_map_type = cal_df[cal_df['type'] == 'pixel_map_type']['misc'].iat[0]
	with PIL.Image.open(calibration_image_path) as calibration_image:
		cam_image_dimensions = np.array(calibration_image.size)
	if cam_image_dimensions[0] > cam_image_dimensions[1]:
		calibration_orientation = 'landscape'
	elif cam_image_dimensions[1] > cam_image_dimensions[0]:
		calibration_orientation = 'portrait'
	elif cam_image_dimensions[0] == cam_image_dimensions[1]:
		calibration_orientation = 'square'

	max_image_index = np.subtract(cam_image_dimensions, 1)
	center_px = max_image_index / 2 # usually xxx.5, non-integer, bc most images have an even number of pixels, means center is a boundary rather than a pixel. Seems most programs accept float values for pixel reference now anyway.
	calibration_quadrant = cal_df[cal_df['type'] == 'quadrant']['misc'].iat[0]
	calibration_distance_cm = float(cal_df[cal_df['type'] == 'distance']['misc'].iat[0])
	align_targets_radius_cm = float(cal_df[cal_df['type'] == 'align_targets_radius']['misc'].iat[0])

	if calibration_orientation == 'portrait':
		center_px = center_px[::-1]
		cam_image_dimensions = cam_image_dimensions[::-1]
		max_image_index = max_image_index[::-1]

	# first for loop: iterate through each calculate_and_record row to determine calibration grid position, correct for errors
	# adjustment 1: the target will be slightly off the center pixel usually.
	# adjustment 2: the grid will be rotated slightly usually. What does the vertical line say the rotation error is?
	# adjustment 3: the grid will be rotated slightly usually. What does the horizontal line say the rotation error is?
	# That's it. Just center and rotate the grid then move on to record the reference points with adjustments.
	for row in cal_df[cal_df['type'] == 'calculate_and_record'].iterrows():
		row_xycm = [float(row[1]['x_cm']), float(row[1]['y_cm'])]
		target_pos_px = cal_df[cal_df['misc']=='target_center'][['x_rec', 'y_rec']].values[0]

		if row[1]['misc'] == 'target_pos_xyangs_relcenter':
			x_small_cm = cal_df[cal_df['misc']=='x_small'][['x_cm', 'y_cm']].values[0]
			x_small_px = cal_df[cal_df['misc']=='x_small'][['x_rec', 'y_rec']].values[0]
			y_small_cm = cal_df[cal_df['misc']=='y_small'][['x_cm', 'y_cm']].values[0]
			y_small_px = cal_df[cal_df['misc']=='y_small'][['x_rec', 'y_rec']].values[0]

			target_pos_xycm_relaim = _target_miss(calibration_distance_cm, center_px, target_pos_px, x_small_cm, x_small_px, y_small_cm, y_small_px)
			cal_df.loc[row[0], 'out1'] = target_pos_xycm_relaim[0]
			cal_df.loc[row[0], 'out2'] = target_pos_xycm_relaim[1]

		elif row[1]['misc'] == 'rotation_error_side_degCCW':
			align_orientation = 'vertical'
			align1_px = cal_df[cal_df['misc']=='align_v_center'][['x_rec', 'y_rec']].values[0]
			align2_px = cal_df[cal_df['misc']=='align_v_left'][['x_rec', 'y_rec']].values[0]
			if not math.isnan(align1_px[0]) and not math.isnan(align1_px[1]) and not math.isnan(align2_px[0]) and not math.isnan(align2_px[1]): # IF NOT BLANK
				grid_rotation_error_degreesCCW = _grid_rotation_error(row_xycm, align_orientation, align1_px, align2_px, align_targets_radius_cm, target_pos_px)
				cal_df.loc[row[0], 'out1'] = grid_rotation_error_degreesCCW
		elif row[1]['misc'] == 'rotation_error_top_degCCW':
			align_orientation = 'horizontal'
			align1_px = cal_df[cal_df['misc']=='align_h_center'][['x_rec', 'y_rec']].values[0]
			align2_px = cal_df[cal_df['misc']=='align_h_down'][['x_rec', 'y_rec']].values[0]
			if not math.isnan(align1_px[0]) and not math.isnan(align1_px[1]) and not math.isnan(align2_px[0]) and not math.isnan(align2_px[1]): # IF NOT BLANK
				grid_rotation_error_degreesCCW = _grid_rotation_error(row_xycm, align_orientation, align1_px, align2_px, align_targets_radius_cm, target_pos_px)
				cal_df.loc[row[0], 'out1'] = grid_rotation_error_degreesCCW
		elif row[1]['misc'] == 'average_rotation_error_degCCW':
			align_v_degCCW = float(cal_df[cal_df['misc']=='rotation_error_side_degCCW'][['out1']].values[0])
			align_h_degCCW = float(cal_df[cal_df['misc']=='rotation_error_top_degCCW'][['out1']].values[0])
			average_rotation_error_degCCW = (align_v_degCCW + align_h_degCCW)/2
			cal_df.loc[row[0], 'out1'] = average_rotation_error_degCCW


	# second for loop: iterate through each reference point, each represented by a "ref_point" row
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
				cal_df.loc[row[0], 'x_px'] = center_px[0]
				cal_df.loc[row[0], 'y_px'] = center_px[1]
				cal_df.loc[row[0], 'xang'] = 0.0
				cal_df.loc[row[0], 'yang'] = 0.0
			elif row_xycm[0]==0 and row_xycm[1]!=0: # on the vertical axis just take the px distance and put on the line with nominal cm value
				px_dist = math.sqrt((px[0] - target_pos_px[0])**2 + (px[1] - target_pos_px[1])**2)
				cal_df.loc[row[0], 'x_px'] = center_px[0]
				cal_df.loc[row[0], 'y_px'] = center_px[1] + px_dist
				yang = math.atan(row_xycm[1]/calibration_distance_cm) * 180/math.pi # (-) as appropriate bc (-) ycm and (+) distance
				cal_df.loc[row[0], 'xang'] = 0.0
				cal_df.loc[row[0], 'yang'] = yang
			elif row_xycm[0]!=0 and row_xycm[1]==0: # on the horizontal axis just take the px distance and put on the line with nominal cm value
				px_dist = math.sqrt((px[0] - target_pos_px[0])**2 + (px[1] - target_pos_px[1])**2)
				cal_df.loc[row[0], 'x_px'] = center_px[0] - px_dist
				cal_df.loc[row[0], 'y_px'] = center_px[1]
				xang = math.atan(row_xycm[0]/calibration_distance_cm) * 180/math.pi # (-) as appropriate bc (-) xcm and (+) distance
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
				r1 = calibration_distance_cm
				r2 = r1 # because the cal board is flat, not a sphere
				yang = math.atan(actual_y_cm_rel_center/r2) * 180/math.pi # mostly (-) because y_cm is (-). the cal board lines would project a small circle on a sphere, so correct
				xang = math.atan(actual_x_cm_rel_center/r1) * 180/math.pi  # is the same as az_rel from diagram bc camera is level and perpendicular to the board
				xyangs = [xang,yang]

				cal_df.loc[row[0], 'x_px'] = px[0]
				cal_df.loc[row[0], 'y_px'] = px[1]
				cal_df.loc[row[0], 'xang'] = xyangs[0]
				cal_df.loc[row[0], 'yang'] = xyangs[1]

	cal_df.to_csv(os.path.join('working', 'output_calibration.csv'))
	ref_df_filter = pd.notnull(cal_df['x_px'])
	ref_df = pd.DataFrame(cal_df[ref_df_filter][['x_px', 'y_px', 'xang', 'yang']])

	# standardize the reference pixels:
	# center, convert to positive values (RB quadrant in PhotoShop coordinate system), then orient landscape
	ref_df['x_px'] = ref_df['x_px'] - center_px[0]
	ref_df['y_px'] = ref_df['y_px'] - center_px[1]
	if calibration_quadrant == 'LB':
		ref_df['x_px'] = ref_df['x_px'] * -1
		ref_df['y_px'] = ref_df['y_px'] * -1 * -1 # twice because y PSpx is (-) down
		ref_df['xang'] = ref_df['xang'] * -1
		ref_df['yang'] = ref_df['yang'] * -1
	if calibration_orientation == 'portrait':
		center_px = center_px[::-1]
		cam_image_dimensions = cam_image_dimensions[::-1]
		max_image_index = max_image_index[::-1]
		ref_df['x_px'], ref_df['y_px'] = ref_df['y_px'], ref_df['x_px']
		ref_df['yang'], ref_df['xang'] = ref_df['xang'], ref_df['yang']

	ref_df.to_csv(os.path.join('working', 'output_refpoints.csv'))

	# create the px to xyangs models
	poly_px = PolynomialFeatures(degree=3, include_bias=False)
	independent_poly_px = pd.DataFrame(data=poly_px.fit_transform(ref_df[['x_px', 'y_px']]), columns=poly_px.get_feature_names_out(ref_df[['x_px', 'y_px']].columns))
	xyangs_model = LinearRegression(fit_intercept=False)
	xyangs_model.fit(independent_poly_px, ref_df[['xang', 'yang']])

	poly_xyangs = PolynomialFeatures(degree=3, include_bias=False)
	independent_poly_xyangs = pd.DataFrame(data=poly_xyangs.fit_transform(ref_df[['xang', 'yang']]), columns=poly_xyangs.get_feature_names_out(ref_df[['xang', 'yang']].columns))
	px_model = LinearRegression(fit_intercept=False)
	px_model.fit(independent_poly_xyangs, ref_df[['x_px', 'y_px']])

	# generate the empty tag and populate it
	cam_AWIMtag = awimlib.generate_empty_AWIMtag_dictionary(default_units=False)
	round_digits = awimlib.AWIMtag_rounding_digits()

	# get some basic exif for completeness
	if calimg_exif_readable.get('Exif.Image.GPSTag'):
		location, location_altitude = awimlib.format_GPS_latlng(calimg_exif_readable)
		
		if location:
			cam_AWIMtag['Location'] = [round(f, round_digits['lat long']) for f in location]
			cam_AWIMtag['LocationSource'] = 'exif GPS'
		else:
			cam_AWIMtag['LocationSource'] = 'Attempted to get from exif GPS, but was not present or not complete.'	
		if location_altitude:
			cam_AWIMtag['LocationAltitude'] = round(location_altitude, round_digits['altitude'])
			cam_AWIMtag['LocationAltitudeSource'] = 'exif GPS'
		else:
			cam_AWIMtag['LocationAltitudeSource'] = 'Attempted to get from exif GPS, but was not present or not complete.'
	else:
		cam_AWIMtag['LocationSource'] = 'Attempted to get from exif GPS, but GPSInfo was not present at all in exif.'
		cam_AWIMtag['LocationAltitudeSource'] = 'Attempted to get from exif GPS, but GPSInfo was not present at all in exif.'
    
	UTC_datetime_str, UTC_source = awimlib.capture_moment_from_exif(calimg_exif_readable)
	if UTC_datetime_str:
		cam_AWIMtag['CaptureMoment'] = UTC_datetime_str
		cam_AWIMtag['CaptureMomentSource'] = UTC_source
	else:
		cam_AWIMtag['CaptureMomentSource'] = 'Attempted to get from exif, but was not present or not complete.'

	# fill in the tag
	cam_AWIMtag['PixelAngleModelsType'] = pixel_map_type

	xyangs_model_df = pd.DataFrame(xyangs_model.coef_, columns=xyangs_model.feature_names_in_, index=['xang_predict', 'yang_predict'])
	# xyangs_model_df.to_csv(os.path.join('working', 'output_xyangs_model_df.csv'))
	xyangs_model_features = list(xyangs_model.feature_names_in_)
	cam_AWIMtag['AnglesModels Features'] = xyangs_model_features
	xang_coeffs = list(xyangs_model_df.loc['xang_predict'].values)
	xang_coeffs = [float(x) for x in xang_coeffs]
	cam_AWIMtag['AnglesModel xang_coeffs'] = xang_coeffs
	yang_coeffs = list(xyangs_model_df.loc['yang_predict'].values)
	yang_coeffs = [float(x) for x in yang_coeffs]
	cam_AWIMtag['AnglesModel yang_coeffs'] = yang_coeffs


	px_model_df = pd.DataFrame(px_model.coef_, columns=px_model.feature_names_in_, index=['x_px_predict', 'y_px_predict'])
	# px_model_df.to_csv(os.path.join('working', 'output_px_model_df.csv'))
	px_model_features = list(px_model.feature_names_in_)
	cam_AWIMtag['PixelsModel Features'] = px_model_features
	xpx_coeffs = list(px_model_df.loc['x_px_predict'].values)
	xpx_coeffs = [float(x) for x in xpx_coeffs]
	cam_AWIMtag['PixelsModel xpx_coeffs'] = xpx_coeffs
	ypx_coeffs = list(px_model_df.loc['y_px_predict'].values)
	ypx_coeffs = [float(x) for x in ypx_coeffs]
	cam_AWIMtag['PixelsModel ypx_coeffs'] = ypx_coeffs

	filler_variable, cam_grid_pxs, cam_TBLR_pxs = awimlib.get_ref_px_and_thirds_grid_TBLR(calibration_image_path, 'center, get from image')
	cam_grid_pxs = cam_grid_pxs.round(round_digits['pixels'])
	cam_grid_angs = awimlib.pxs_to_xyangs(cam_AWIMtag, cam_grid_pxs)
	cam_grid_angs = cam_grid_angs.round(round_digits['degrees'])
	cam_TBLR_pxs = cam_TBLR_pxs.round(round_digits['pixels'])
	cam_TBLR_angs = awimlib.pxs_to_xyangs(cam_AWIMtag, cam_TBLR_pxs)
	cam_TBLR_angs = cam_TBLR_angs.round(round_digits['degrees'])

	cam_AWIMtag['GridPixels'] = cam_grid_pxs.tolist()
	cam_AWIMtag['GridAngles'] = cam_grid_angs.tolist()
	cam_AWIMtag['TBLRPixels'] = cam_TBLR_pxs.tolist()
	cam_AWIMtag['TBLRAngles'] = cam_TBLR_angs.tolist()

	px_size_center, px_size_average = awimlib.get_pixel_sizes(calibration_image_path, cam_AWIMtag)
	cam_AWIMtag['PixelSizeCenterHorizontalVertical'] = [round(f, round_digits['pixels']) for f in px_size_center]
	cam_AWIMtag['PixelSizeAverageHorizontalVertical'] = [round(f, round_digits['pixels']) for f in px_size_average]
	cam_AWIMtag['PixelSizeUnit'] = 'Pixels per Degree'

	cam_AWIMtag_jsonready = awimlib.dict_json_ready(cam_AWIMtag)
	file_path = os.path.join('working', 'output_cal {} {} awim.json'.format(camera_name, lens_name))
	with open(file_path, 'w') as text_file:
		json.dump(cam_AWIMtag_jsonready, text_file, indent=4)

	return True


def represent_camera(self):
	camera_str = ''
	camera_str += '\nCamera Name: ' + self.camera_name
	camera_str += '\nLens: '+self.lens_name
	camera_str += '\nZoom: '+self.zoom_factor
	camera_str += '\nSettings Notes: '+self.settings_notes
	camera_str += '\nImage Dimensions: %i, %i' % (self.cam_image_dimensions[0], self.cam_image_dimensions[1])
	camera_str += '\n\nPixel Corners LT, RT, LB, RB:\n' + ', '.join(str(x) for x in self.px_corners)
	camera_str += '\nx,y Angle Corners LT, RT, LB, RB:\n' + ', '.join(str(x) for x in self.xyangs_corners)
	camera_str += '\n\nPixel Edges Top, Right, Bottom, Left:\n' + ', '.join(str(x) for x in self.px_edges)
	camera_str += '\nx,y Angle Edges Top, Right, Bottom, Left:\n' + ', '.join(str(x) for x in self.xyangs_edges)
	camera_str += '\n\nDegrees per hundred pixels horizontal, avg full image: %.2f' % (100 * (2*self.xyangs_edges[1,0]) / self.cam_image_dimensions[0])
	camera_str += '\nDegrees per hundred pixels vertical, avg full image: %.2f' % (100 * (2*self.xyangs_edges[0,1]) / self.cam_image_dimensions[1])
	
	print(camera_str)
	with open(r'code-output-dump-folder/camera awim data.txt', 'w') as f:
		f.write(camera_str)


	def generate_xyang_pixel_models(self, src_img_path, img_orientation, img_tilt):
		source_image = Image.open(src_img_path)
		img_dimensions = source_image.size
		if img_orientation == 'landscape':
			cam_dimensions = [self.cam_image_dimensions[0], self.cam_image_dimensions[1]]
		elif img_orientation == 'portrait':
			cam_dimensions = [self.cam_image_dimensions[1], self.cam_image_dimensions[0]]
		img_aspect_ratio = img_dimensions[0] / img_dimensions[1]
		cam_aspect_ratio = cam_dimensions[0] / cam_dimensions[1]

		if abs(img_aspect_ratio - cam_aspect_ratio) > 0.001:
			print('error: image aspect ratio does not match camera aspect ratio, but it should. Was the image cropped? Not supported yet.')
		else:
			img_resize_factor = img_dimensions[0] / cam_dimensions[0]

		if img_orientation == 'landscape':
			xyangs_model_df = pd.DataFrame(self.xyangs_model.coef_, columns=self.xyangs_model.feature_names_in_, index=['xang_predict', 'yang_predict'])
			px_model_df = pd.DataFrame(self.px_model.coef_, columns=self.px_model.feature_names_in_, index=['x_px_predict', 'y_px_predict'])
		elif img_orientation == 'portrait': # TODO: make this the transpose of landscape / swap x and y
			xyangs_model_df = pd.DataFrame(self.xyangs_model.coef_, columns=self.xyangs_model.feature_names_in_, index=['xang_predict', 'yang_predict'])
			px_model_df = pd.DataFrame(self.px_model.coef_, columns=self.px_model.feature_names_in_, index=['x_px_predict', 'y_px_predict'])
		xyangs_model_df /= img_resize_factor
		px_model_df *= img_resize_factor

		return self.pixel_map_type, xyangs_model_df, px_model_df


	def px_xyangs_models_convert(self, input, direction):
		if isinstance(input, list): # models require numpy arrays
			input = np.array(input, dtype=float).reshape(-1,2)

		full_shape = input.shape # save the shape for reshape later
		output = np.zeros(full_shape)
		output_sign = np.where(input < 0, -1, 1) # models are positive values only. Save sign.
		input = np.abs(input)
		input = input.reshape(-1,2)
		if direction == 'px_to_xyangs':
			input_poly = self.poly_px.fit_transform(input)
			output = self.xyangs_model.predict(input_poly).reshape(full_shape)
		elif direction == 'xyangs_to_px':
			input_poly = self.poly_xyangs.fit_transform(input)
			output = self.px_model.predict(input_poly).reshape(full_shape)
		output = np.multiply(output, output_sign)

		return output


# Apparently I stopped using this.
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