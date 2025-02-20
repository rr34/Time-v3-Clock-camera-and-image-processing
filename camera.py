import math
import numpy as np
import pandas as pd
import os
import PIL
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import awimlib, metadata_tools, formatters, astropytools


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


# calculate the miss angle as xang,yang
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


# works 30 Jan 2025
def generate_camera_AWIM_from_calibration(calibration_image_path, calibration_file_path):
	calimg_exif_dict = metadata_tools.get_metadata(calibration_image_path)
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
	cam_AWIMtag = awimlib.generate_empty_AWIMtag_dictionary()

	# get some basic exif for completeness
	if calimg_exif_dict.get('Exif.Image.GPSTag'):
		location, location_altitude = formatters.format_GPS_latlng(calimg_exif_dict)
		
		if location:
			cam_AWIMtag['awim Location Coordinates'] = location
			cam_AWIMtag['awim Location Coordinates Source'] = 'exif GPS'
		else:
			cam_AWIMtag['awim Location Coordinates Source'] = 'Attempted to get from exif GPS, but was not present or not complete.'	
		if location_altitude:
			cam_AWIMtag['awim Location MSL'] = location_altitude
			cam_AWIMtag['awim Location MSL Source'] = 'exif GPS'
		else:
			cam_AWIMtag['awim Location MSL Source'] = 'Attempted to get from exif GPS, but was not present or not complete.'
	else:
		cam_AWIMtag['awim Location Coordinates Source'] = 'Attempted to get from exif GPS, but GPSInfo was not present at all in exif.'
		cam_AWIMtag['awim Location MSL Source'] = 'Attempted to get from exif GPS, but GPSInfo was not present at all in exif.'
    
	UTC_datetime_str, UTC_source = metadata_tools.capture_moment_from_metadata(calimg_exif_dict)
	if UTC_datetime_str:
		cam_AWIMtag['awim Capture Moment'] = UTC_datetime_str
		cam_AWIMtag['awim Capture Moment Source'] = UTC_source
	else:
		cam_AWIMtag['awim Capture Moment Source'] = 'Attempted to get from exif, but was not present or not complete.'

	# fill in the tag
	cam_AWIMtag['awim Ref Image Size'] = cam_image_dimensions.tolist()
	cam_AWIMtag['awim Models Type'] = pixel_map_type

	xyangs_model_df = pd.DataFrame(xyangs_model.coef_, columns=xyangs_model.feature_names_in_, index=['xang_predict', 'yang_predict'])
	# xyangs_model_df.to_csv(os.path.join('working', 'output_xyangs_model_df.csv'))
	xyangs_model_features = list(xyangs_model.feature_names_in_)
	cam_AWIMtag['awim Angles Models Features'] = xyangs_model_features
	xang_coeffs = list(xyangs_model_df.loc['xang_predict'].values)
	xang_coeffs = [float(x) for x in xang_coeffs]
	cam_AWIMtag['awim Angles Model xang_coeffs'] = xang_coeffs
	yang_coeffs = list(xyangs_model_df.loc['yang_predict'].values)
	yang_coeffs = [float(x) for x in yang_coeffs]
	cam_AWIMtag['awim Angles Model yang_coeffs'] = yang_coeffs


	px_model_df = pd.DataFrame(px_model.coef_, columns=px_model.feature_names_in_, index=['x_px_predict', 'y_px_predict'])
	# px_model_df.to_csv(os.path.join('working', 'output_px_model_df.csv'))
	px_model_features = list(px_model.feature_names_in_)
	cam_AWIMtag['awim Pixels Model Features'] = px_model_features
	xpx_coeffs = list(px_model_df.loc['x_px_predict'].values)
	xpx_coeffs = [float(x) for x in xpx_coeffs]
	cam_AWIMtag['awim Pixels Model xpx_coeffs'] = xpx_coeffs
	ypx_coeffs = list(px_model_df.loc['y_px_predict'].values)
	ypx_coeffs = [float(x) for x in ypx_coeffs]
	cam_AWIMtag['awim Pixels Model ypx_coeffs'] = ypx_coeffs

	ref_px, cam_grid_pxs, cam_TBLR_pxs = awimlib.get_ref_px_thirds_grid_TBLR(calibration_image_path, 'center, get from image')
	cam_grid_angs = awimlib.pxs_to_xyangs(cam_AWIMtag, cam_grid_pxs)
	cam_TBLR_angs = awimlib.pxs_to_xyangs(cam_AWIMtag, cam_TBLR_pxs)

	cam_AWIMtag['awim Ref Pixel'] = ref_px
	cam_AWIMtag['awim Grid Pixels'] = cam_grid_pxs.tolist()
	cam_AWIMtag['awim Grid Angles'] = cam_grid_angs.tolist()
	cam_AWIMtag['awim TBLR Pixels'] = cam_TBLR_pxs.tolist()
	cam_AWIMtag['awim TBLR Angles'] = cam_TBLR_angs.tolist()

	px_size_center, px_size_average = awimlib.get_pixel_sizes(cam_AWIMtag)
	cam_AWIMtag['awim Pixel Size Center Horizontal Vertical'] = px_size_center
	cam_AWIMtag['awim Pixel Size Average Horizontal Vertical'] = px_size_average

	cam_AWIMtag = formatters.round_AWIMtag(cam_AWIMtag)

	filename = 'output_cal {} {} cam_awim.json'.format(camera_name, lens_name)

	cam_AWIMtag.update(calimg_exif_dict)

	return cam_AWIMtag, filename


def generate_tag_from_exif_plus_misc(image_path, cam_AWIMtag_dictionary, photoshoot_dictionary):
	# AWIMtag_dictionary['Location'] = [40.298648, -83.055772] # Time v3 Technology shop default for now.
	# tz = 'US/Eastern'
	metadata_dict = metadata_tools.get_metadata(image_path)
	AWIMtag_dictionary = awimlib.generate_empty_AWIMtag_dictionary()
	with PIL.Image.open(image_path) as image:
		image_dimensions = np.array(image.size)


	if 'some user selection variable' == 'try camera gps': # todo: use GPS from camera if present, usually not in my case.
		gps_location = metadata_tools.GPS_location_from_metadata(metadata_dict)
		AWIMtag_dictionary['awim Location Coordinates'] = gps_location
		AWIMtag_dictionary['awim Location Coordinates Source'] = 'Camera GPS'
	elif True: # use location from the photoshoot dictionary
		latlng = photoshoot_dictionary['LatLong'].split(',')
		latlng = [float(latlng[0]), float(latlng[1])]
		AWIMtag_dictionary['awim Location Coordinates'] = latlng
		AWIMtag_dictionary['awim Location Coordinates Source'] = 'Photoshoot Data'

	AWIMtag_dictionary['awim Location MSL'] = photoshoot_dictionary['PhotoMSL'] # todo: check for GPS MSL
	AWIMtag_dictionary['awim Location MSL Source'] = 'Photoshoot Data'

	AWIMtag_dictionary['awim Location Terrain Elevation'] = photoshoot_dictionary['TerrainElevation']
	AWIMtag_dictionary['awim Location Terrain Elevation Source'] = 'Photoshoot Data'

	AWIMtag_dictionary['awim Location AGL'] = photoshoot_dictionary['PhotoAGL'] # todo: use to compare MSL and terrain elevation
	AWIMtag_dictionary['awim Location AGL Description'] = photoshoot_dictionary['PhotoAGLDescription']
	AWIMtag_dictionary['awim Location AGL Source'] = 'Photoshoot Data'

	datetime_str, datetime_source = metadata_tools.capture_moment_from_metadata(metadata_dict)
	camtime_adjustment = photoshoot_dictionary['CamTimeError']
	datetime_str = formatters.adjust_datetime_byseconds(datetime_str, camtime_adjustment)
	AWIMtag_dictionary['awim Capture Moment'] = datetime_str
	AWIMtag_dictionary['awim Capture Moment Source'] = datetime_source

	AWIMtag_dictionary['awim Models Type'] = cam_AWIMtag_dictionary['awim Models Type']

	ref_px, img_grid_pxs, img_TBLR_pxs = awimlib.get_ref_px_thirds_grid_TBLR(image_path, 'center, get from image')
	AWIMtag_dictionary['awim Ref Pixel'] = ref_px

	AWIMtag_dictionary['awim Ref Image Size'] = cam_AWIMtag_dictionary['awim Ref Image Size']
	AWIMtag_dictionary['awim Angles Models Features'] = cam_AWIMtag_dictionary['awim Angles Models Features']
	AWIMtag_dictionary['awim Angles Model xang_coeffs'] = cam_AWIMtag_dictionary['awim Angles Model xang_coeffs']
	AWIMtag_dictionary['awim Angles Model yang_coeffs'] = cam_AWIMtag_dictionary['awim Angles Model yang_coeffs']

	AWIMtag_dictionary['awim Pixels Model Features'] = cam_AWIMtag_dictionary['awim Pixels Model Features']
	AWIMtag_dictionary['awim Pixels Model xpx_coeffs'] = cam_AWIMtag_dictionary['awim Pixels Model xpx_coeffs']
	AWIMtag_dictionary['awim Pixels Model ypx_coeffs'] = cam_AWIMtag_dictionary['awim Pixels Model ypx_coeffs']

	img_orientation = photoshoot_dictionary['Orientation']
	azart_source = photoshoot_dictionary['AzArtSource']

	# AzArt option 1 ... of several
	if azart_source == 'az offset from reference':
		artifae = photoshoot_dictionary['Artifae']
		ref_az = photoshoot_dictionary['RefAz'] # this is the guess direction of the reference object, not of the photo direction
		obj_type = photoshoot_dictionary['ObjAzType']
		# get a reference azimuth from an object or from shooting an azimuth between two points
		if azart_source == 'two points':
			latlng1 = photoshoot_dictionary['LatLongPt1']
			latlng2 = photoshoot_dictionary['LatLongPt2']
			obj_az = 'get azimuth from the two lat long coordinates'
		elif azart_source != 'two points':
			obj_az = photoshoot_dictionary['ObjAz']
		if obj_type == 'rectangle':
			obj_sides = 4
		elif obj_type in ('halves', 'two points'):
			obj_sides = 2
		obj_az = awimlib.closest_to_x_sides(ref_az, obj_az, obj_sides) # adjusts object azimuth to the actual measured azimuth of the object using a close enough guess

		# adjust the photo azimuth from the reference azimuth using tripod readings
		tripod1 = photoshoot_dictionary['RefTripod1']
		tripod2 = photoshoot_dictionary['RefTripod2']
		angle_moved = abs(tripod2 - tripod1)
		tripod_direction = photoshoot_dictionary['RefTripodDirectionMoved']
		if tripod_direction == 'left':
			sign = -1
		elif tripod_direction == 'right':
			sign = 1
		azimuth = obj_az + sign*angle_moved
		azimuth = (azimuth + 360) % 360
		azart = [azimuth, artifae]
		AWIMtag_dictionary['awim Ref Pixel Azimuth Artifae'] = azart
		AWIMtag_dictionary['awim Ref Pixel Azimuth Artifae Source'] = 'Adjustment from reference object using tripod readings.'
	# AzArt option 2 ...
	elif azart_source == 'option 2, celestial object':
		# if AWIMtag_dictionary['awim Azimuth Artifae Source'] == 'from known px':
		# 	if isinstance(known_px_azart, str):
		# 		ref_azart_source = 'From celestial object in photo: ' + known_px_azart
		# 		known_px_azart = astropytools.get_AzArt(AWIMtag_dictionary, known_px_azart)

		# 	ref_azart = ref_px_from_known_px(AWIMtag_dictionary, known_px, known_px_azart)
		# AWIMtag_dictionary['awim Ref Pixel Azimuth Artifae'] = ref_azart.round(round_digits['degrees'])
		# AWIMtag_dictionary['awim Ref Pixel Azimuth Artifae Source'] = ref_azart_source
		pass

	# get grid angles, azimuth artifae, RA Dec. Grid pixels from above. Unless cropped, should be the same as the camera
	AWIMtag_dictionary['awim Grid Pixels'] = img_grid_pxs.tolist()
	grid_angs = awimlib.pxs_to_xyangs(AWIMtag_dictionary, img_grid_pxs)
	AWIMtag_dictionary['awim Grid Angles'] = grid_angs.tolist()

	grid_azarts = awimlib.xyangs_to_azarts(AWIMtag_dictionary, grid_angs)
	AWIMtag_dictionary['awim Grid Azimuth Artifae'] = grid_azarts.tolist()

	image_moment = AWIMtag_dictionary['awim Capture Moment']
	image_location = AWIMtag_dictionary['awim Location Coordinates']
	grid_RADecs = astropytools.AzArts_to_RADecs(image_location, image_moment, grid_azarts)
	AWIMtag_dictionary['awim Grid RA Dec'] = grid_RADecs.tolist()

	# get top, bottom, left, right (TBLR) angles, azimuth artifae, RA Dec. TBLR pixels from above. Unless cropped, should be the same as the camera
	AWIMtag_dictionary['awim TBLR Pixels'] = img_TBLR_pxs.tolist()
	TBLR_angs = awimlib.pxs_to_xyangs(AWIMtag_dictionary, img_TBLR_pxs)
	AWIMtag_dictionary['awim TBLR Angles'] = TBLR_angs.tolist()

	TBLR_azarts = awimlib.xyangs_to_azarts(AWIMtag_dictionary, TBLR_angs)
	AWIMtag_dictionary['awim TBLR Azimuth Artifae'] = TBLR_azarts.tolist()

	image_moment = AWIMtag_dictionary['awim Capture Moment']
	image_location = AWIMtag_dictionary['awim Location Coordinates']
	TBLR_RADecs = astropytools.AzArts_to_RADecs(image_location, image_moment, TBLR_azarts)
	AWIMtag_dictionary['awim TBLR RA Dec'] = TBLR_RADecs.tolist()

	px_size_center, px_size_average = awimlib.get_pixel_sizes(AWIMtag_dictionary)
	AWIMtag_dictionary['awim Pixel Size Center Horizontal Vertical'] = px_size_center
	AWIMtag_dictionary['awim Pixel Size Average Horizontal Vertical'] = px_size_average

	AWIMtag_dictionary = formatters.round_AWIMtag(AWIMtag_dictionary)

	AWIMtag_dictionary.update(metadata_dict)

	return AWIMtag_dictionary