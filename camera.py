import math
import numpy as np
import pandas as pd
from PIL import Image, PngImagePlugin
import pickle
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
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
		camera_name = cal_df[cal_df['type'] == 'camera_name']['misc'].iat[0]
		lens_name = cal_df[cal_df['type'] == 'lens_name']['misc'].iat[0]
		zoom_factor = cal_df[cal_df['type'] == 'zoom_factor']['misc'].iat[0]
		settings_notes = cal_df[cal_df['type'] == 'settings_notes']['misc'].iat[0]
		sensor_dimensions = [int(cal_df[cal_df['type'] == 'sensor_dimensions']['rec_x'].iat[0]), int(cal_df[cal_df['type'] == 'sensor_dimensions']['rec_y'].iat[0])]
		max_sensor_index = np.subtract(sensor_dimensions, 1)
		cal_image_dimensions = [int(cal_df[cal_df['type'] == 'cal_image_dimensions']['rec_x'].iat[0]), int(cal_df[cal_df['type'] == 'cal_image_dimensions']['rec_y'].iat[0])]
		max_image_index = np.subtract(cal_image_dimensions, 1)
		pixel_map_type = cal_df[cal_df['type'] == 'pixel_map_type']['misc'].iat[0]
		center_px = max_image_index / 2 # usually x.5, non-integer, bc even number of pixels means center is a boundary rather than a pixel
		calibration_orientation = cal_df[cal_df['type'] == 'orientation']['misc'].iat[0]
		targets_diameter_degrees = float(cal_df[cal_df['type'] == 'targets_diameter']['misc'].iat[0])
		calibration_quadrant = cal_df[cal_df['type'] == 'quadrant']['misc'].iat[0]


		if calibration_orientation == 'portrait':
			center_px = center_px[::-1]
			sensor_dimensions = sensor_dimensions[::-1]
			max_sensor_index = max_sensor_index[::-1]
			cal_image_dimensions = cal_image_dimensions[::-1]
			max_image_index = max_image_index[::-1]



		# first for loop: each calc step represented by a "calculate_and_record" row: get vars, check if there, handle cases
		# crosshairs case 1: record center as original crosshairs
		# crosshairs case 2. on az axis (vertical)
		# crosshairs case 3. on alt axis (horizontal)
		# crosshairs case 4. not on an axis, "floating"
		# target miss: calculate the target position azalt relative to the original crosshairs, used for all ref points with same az/alt _sect
		# NOT USED, calculate the target center from edges just to compare to the actual recorded center, which is used
		# rotation error: (°CCW) from the align points, used for all ref points with same az/alt _sect
		# pixel delta: °/px used for adjusting pixel location (instead of azalt) for "floating" crosshairs
		# IF NOT BLANK marks the "check for values" if statement
		for row in cal_df[cal_df['type'] == 'calculate_and_record'].iterrows():
			# prepare variables
			azalt_sect = row[1][['az_sect', 'alt_sect']].values
			azalt_rel = row[1][['az_rel', 'alt_rel']].values
			azalt_total = [azalt_sect[0]+azalt_rel[0], azalt_sect[1]+azalt_rel[1]]
			cal_df.loc[row[0], 'az_total'] = azalt_total[0]
			cal_df.loc[row[0], 'alt_total'] = azalt_total[1]

			df_find_rows_section = (cal_df['az_sect']==row[1]['az_sect']) & (cal_df['alt_sect']==row[1]['alt_sect'])
			df_section = cal_df[df_find_rows_section]
			df_find_rows_pt = (cal_df['az_sect']==row[1]['az_sect']) & (cal_df['alt_sect']==row[1]['alt_sect']) & (cal_df['az_rel']==row[1]['az_rel']) & (cal_df['alt_rel']==row[1]['alt_rel'])
			df_pt = cal_df[df_find_rows_pt]

			df_find_row_target = (df_section['misc']=='target_center')
			target_center_section = df_section[df_find_row_target][['rec_x', 'rec_y']].values[0]

			df_find_crosshairs_sect = (cal_df['az_total']==azalt_sect[0]) & (cal_df['alt_total']==azalt_sect[1]) & (cal_df['type']=='calculate_and_record') & (cal_df['misc']=='this_section_crosshairs')
			crosshairs_section = cal_df[df_find_crosshairs_sect][['out1', 'out2']].values[0]

			if row[1]['misc'] == 'this_section_crosshairs':
				if azalt_total[0]==0 and azalt_total[1]==0: # origin only
					cal_df.loc[row[0], 'out1'] = center_px[0]
					cal_df.loc[row[0], 'out2'] = center_px[1]
				else:
					# retrieve crosshairs from previous section(s)
					df_find_crosshairs = (cal_df['az_total']==azalt_total[0]) & (cal_df['alt_total']==azalt_total[1]) & (cal_df['type']=='calculate_and_record') & (cal_df['misc']=='crosshairs')
					crosshairs_array = cal_df[df_find_crosshairs][['out1', 'out2']]
					not_blank, crosshairs = coord_standard(crosshairs_array)
					if not_blank: # IF NOT BLANK
						if isinstance(crosshairs, np.ndarray):
							crosshairs1 = crosshairs[0,:]
							crosshairs2 = crosshairs[1,:]
							crosshairs = [(crosshairs1[0]+crosshairs2[0])/2, (crosshairs1[1]+crosshairs2[1])/2]
						cal_df.loc[row[0], 'out1'] = crosshairs[0]
						cal_df.loc[row[0], 'out2'] = crosshairs[1]

			elif row[1]['misc'] == 'crosshairs':
				df_find_recorded = (df_pt['misc']=='center') & (df_pt['type']=='ref_point')
				px = df_pt[df_find_recorded][['rec_x', 'rec_y']]
				not_blank, px = coord_standard(px)
				# if a px is recorded, case 2 on az axis, case 3 on alt axis, case 4 on neither axis
				if not_blank: # IF NOT BLANK
					if azalt_total[0]==0:
						px_dist = math.sqrt((px[0] - target_center_section[0])**2 + (px[1] - target_center_section[1])**2)
						crosshairs_px = [center_px[0], crosshairs_section[1] + px_dist]
						cal_df.loc[row[0], 'out1'] = crosshairs_px[0]
						cal_df.loc[row[0], 'out2'] = crosshairs_px[1]
					elif azalt_total[1]==0:
						px_dist = math.sqrt((px[0] - target_center_section[0])**2 + (px[1] - target_center_section[1])**2)
						crosshairs_px = [crosshairs_section[0] - px_dist, center_px[1]]
						cal_df.loc[row[0], 'out1'] = crosshairs_px[0]
						cal_df.loc[row[0], 'out2'] = crosshairs_px[1]
					else: # adjust the pixel for crosshairs, not the azalt
						df_find_row_tpaara = (df_section['misc']=='target_pos_azalt_relaim')
						target_pos_azalt_relaim = df_section[df_find_row_tpaara][['out1', 'out2']].values[0]
						df_find_row_pdaa_v = (df_section['misc']=='rotation_error_v_and_px_delta')
						pixel_delta_at_align_v = df_section[df_find_row_pdaa_v][['out2']].values[0]
						df_find_row_pdaa_h = (df_section['misc']=='rotation_error_h_and_px_delta')
						pixel_delta_at_align_h = df_section[df_find_row_pdaa_h][['out2']].values[0]
						# note there is no grid rotation error for these cases because not on an axis
						crosshairs_px = np.array([1,2])
						crosshairs_px[0] = px[0] + pixel_delta_at_align * target_pos_azalt_relaim[0]
						crosshairs_px[1] = px[1] - pixel_delta_at_align * target_pos_azalt_relaim[1]
						cal_df.loc[row[0], 'out1'] = crosshairs_px[0]
						cal_df.loc[row[0], 'out2'] = crosshairs_px[1]

			elif row[1]['misc'] == 'target_pos_azalt_relaim':
				target_top = df_pt[df_pt['misc']=='target_top'][['rec_x', 'rec_y']].values[0]
				target_right = df_pt[df_pt['misc']=='target_right'][['rec_x', 'rec_y']].values[0]
				target_bottom = df_pt[df_pt['misc']=='target_bottom'][['rec_x', 'rec_y']].values[0]
				target_left = df_pt[df_pt['misc']=='target_left'][['rec_x', 'rec_y']].values[0]

				target_pos_azalt_relaim = target_miss(target_left, target_right, target_top, target_bottom, target_center_section, targets_diameter_degrees, crosshairs_section)
				cal_df.loc[row[0], 'out1'] = target_pos_azalt_relaim[0]
				cal_df.loc[row[0], 'out2'] = target_pos_azalt_relaim[1]

			elif row[1]['misc'] == 'target_center_calculated_from_edges':
				target_top = df_pt[df_pt['misc']=='target_top'][['rec_x', 'rec_y']].values[0]
				target_right = df_pt[df_pt['misc']=='target_right'][['rec_x', 'rec_y']].values[0]
				target_bottom = df_pt[df_pt['misc']=='target_bottom'][['rec_x', 'rec_y']].values[0]
				target_left = df_pt[df_pt['misc']=='target_left'][['rec_x', 'rec_y']].values[0]
				if not math.isnan(target_top[0]): # IF NOT BLANK
					target_pos_calculated_notused = intersect_two_lines(target_left, target_right, target_top, target_bottom)
					cal_df.loc[row[0], 'out1'] = target_pos_calculated_notused[0]
					cal_df.loc[row[0], 'out2'] = target_pos_calculated_notused[1]

			elif row[1]['misc'] == 'rotation_error_v_and_px_delta':
				align_orientation = 'vertical'
				align1_px = df_section[df_section['misc']=='vertical_left'][['rec_x', 'rec_y']].values[0]
				align2_px = df_section[df_section['misc']=='vertical_right'][['rec_x', 'rec_y']].values[0]
				if not math.isnan(align1_px[0]): # IF NOT BLANK
					grid_rotation_error_degreesCCW, pixel_delta_at_align = grid_rotation_error(align_orientation=align_orientation, align1_px=align1_px, align2_px=align2_px, align_azalt_reltarget=azalt_rel, targets_diameter_degrees=targets_diameter_degrees, target_pos_px=target_center_section)
					cal_df.loc[row[0], 'out1'] = grid_rotation_error_degreesCCW
					cal_df.loc[row[0], 'out2'] = pixel_delta_at_align

			elif row[1]['misc'] == 'rotation_error_h_and_px_delta':
				align_orientation = 'horizontal'
				align1_px = df_section[df_section['misc']=='horizontal_top'][['rec_x', 'rec_y']].values[0]
				align2_px = df_section[df_section['misc']=='horizontal_bottom'][['rec_x', 'rec_y']].values[0]
				if not math.isnan(align1_px[0]): # IF NOT BLANK
					grid_rotation_error_degreesCCW, pixel_delta_at_align = grid_rotation_error(align_orientation=align_orientation, align1_px=align1_px, align2_px=align2_px, align_azalt_reltarget=azalt_rel, targets_diameter_degrees=targets_diameter_degrees, target_pos_px=target_center_section)
					cal_df.loc[row[0], 'out1'] = grid_rotation_error_degreesCCW
					cal_df.loc[row[0], 'out2'] = pixel_delta_at_align



		# second for loop: each reference point represented by a "ref_point" row
		# ref point case 1: record the center as a reference point bc defining as original azalt = [0,0]
		# ref point case 2: like crosshairs case 2, on az axis (vertical)
		# ref point case 3: like crosshairs case 3. on alt axis (horizontal)
		# ref point case 4: normal ref point
		for row in cal_df[cal_df['type'] == 'ref_point'].iterrows():
			# prepare variables
			px = row[1][['rec_x', 'rec_y']].values
			azalt_sect = row[1][['az_sect', 'alt_sect']].values
			azalt_rel = row[1][['az_rel', 'alt_rel']].values
			azalt_total = [azalt_sect[0]+azalt_rel[0], azalt_sect[1]+azalt_rel[1]]
			cal_df.loc[row[0], 'az_total'] = azalt_total[0]
			cal_df.loc[row[0], 'alt_total'] = azalt_total[1]

			df_find_rows_section = (cal_df['az_sect']==row[1]['az_sect']) & (cal_df['alt_sect']==row[1]['alt_sect'])
			df_section = cal_df[df_find_rows_section]
			df_find_rows_pt = (cal_df['az_sect']==row[1]['az_sect']) & (cal_df['alt_sect']==row[1]['alt_sect']) & (cal_df['az_rel']==row[1]['az_rel']) & (cal_df['alt_rel']==row[1]['alt_rel'])
			df_pt = cal_df[df_find_rows_pt]

			df_find_row_target = (df_section['misc']=='target_center')
			target_center_section = df_section[df_find_row_target][['rec_x', 'rec_y']].values[0]

			df_find_crosshairs_sect = (cal_df['az_total']==azalt_sect[0]) & (cal_df['alt_total']==azalt_sect[1]) & (cal_df['type']=='calculate_and_record') & (cal_df['misc']=='this_section_crosshairs')
			crosshairs_section = cal_df[df_find_crosshairs_sect][['out1', 'out2']].values[0]

			if not math.isnan(px[0]) and not math.isnan(px[1]): # IF NOT BLANK
				# get the grid rotation error variable(s) from the dataframe and either:
				# average two, just vertical, just horizontal, or none
				df_gredCCW_pd_v = df_section[df_section['misc']=='rotation_error_v_and_px_delta']
				grid_rotation_error_degreesCCW_v = float(df_gredCCW_pd_v[['out1']].values[0])
				pixel_delta_v = float(df_gredCCW_pd_v[['out2']].values[0])
				df_gredCCW_pd_h = df_section[df_section['misc']=='rotation_error_h_and_px_delta']
				grid_rotation_error_degreesCCW_h = float(df_gredCCW_pd_h[['out1']].values[0])
				pixel_delta_h = float(df_gredCCW_pd_h[['out2']].values[0])
				if azalt_sect[0]==0 and azalt_sect[1]==0 and not math.isnan(grid_rotation_error_degreesCCW_v) and not math.isnan(grid_rotation_error_degreesCCW_h):
					grid_rotation_error_degreesCCW = (grid_rotation_error_degreesCCW_v + grid_rotation_error_degreesCCW_h) / 2
				elif azalt_sect[0]==0 and azalt_sect[1]!=0 and not math.isnan(grid_rotation_error_degreesCCW_v):
					grid_rotation_error_degreesCCW = grid_rotation_error_degreesCCW_v
				elif azalt_sect[0]!=0 and azalt_sect[1]==0 and not math.isnan(grid_rotation_error_degreesCCW_h):
					grid_rotation_error_degreesCCW = grid_rotation_error_degreesCCW_h
				else:
					grid_rotation_error_degreesCCW = 0 # CATCH-ALL FOR BLANKS
			
				# record the ref pixels for the 4 cases
				if azalt_total[0]==0 and azalt_total[1]==0: # origin
					cal_df.loc[row[0], 'px_x'] = center_px[0]
					cal_df.loc[row[0], 'px_y'] = center_px[1]
					cal_df.loc[row[0], 'az'] = 0
					cal_df.loc[row[0], 'alt'] = 0
				elif azalt_total[0]==0 and azalt_total[1]!=0:
					px_dist = math.sqrt((px[0] - target_center_section[0])**2 + (px[1] - target_center_section[1])**2)
					cal_df.loc[row[0], 'px_x'] = center_px[0]
					cal_df.loc[row[0], 'px_y'] = crosshairs_section[1] + px_dist
					cal_df.loc[row[0], 'az'] = azalt_total[0]
					cal_df.loc[row[0], 'alt'] = azalt_total[1]
				elif azalt_total[0]!=0 and azalt_total[1]==0:
					px_dist = math.sqrt((px[0] - target_center_section[0])**2 + (px[1] - target_center_section[1])**2)
					cal_df.loc[row[0], 'px_x'] = crosshairs_section[0] - px_dist
					cal_df.loc[row[0], 'px_y'] = center_px[1]
					cal_df.loc[row[0], 'az'] = azalt_total[0]
					cal_df.loc[row[0], 'alt'] = azalt_total[1]
				else:
					target_pos_azalt_relaim = df_section[(df_section['misc']=='target_pos_azalt_relaim')][['out1', 'out2']].values[0]
					theta, r = azalt_to_special_polar(azalt_total[0], azalt_total[1])
					grid_rotation_adjust_azalt_relgridpoint =  [math.tan(grid_rotation_error_degreesCCW*math.pi/180) * r * math.cos(theta*math.pi/180) * -1, math.tan(grid_rotation_error_degreesCCW*math.pi/180) * r * math.sin(theta*math.pi/180)]
					cal_df.loc[row[0], 'px_x'] = px[0]
					cal_df.loc[row[0], 'px_y'] = px[1]
					cal_df.loc[row[0], 'az'] = azalt_total[0] + target_pos_azalt_relaim[0] + grid_rotation_adjust_azalt_relgridpoint[0]
					cal_df.loc[row[0], 'alt'] = azalt_total[1] + target_pos_azalt_relaim[1] + grid_rotation_adjust_azalt_relgridpoint[1]

		cal_df.to_csv('slay cal output.csv')
		ref_df_filter = pd.notnull(cal_df['px_x'])
		ref_df = pd.DataFrame(cal_df[ref_df_filter][['px_x', 'px_y', 'az', 'alt']])

		# standardize the reference pixels:
		# center, convert to positive values (RB quadrant in PhotoShop coordinate system), then orient landscape
		ref_df['px_x'] = ref_df['px_x'] - center_px[0]
		ref_df['px_y'] = ref_df['px_y'] - center_px[1]
		if calibration_quadrant == 'LB':
			ref_df['px_x'] = ref_df['px_x'] * -1
			ref_df['az'] = ref_df['az'] * -1
			ref_df['alt'] = ref_df['alt'] * -1
		if calibration_orientation == 'portrait':
			center_px = center_px[::-1]
			sensor_dimensions = sensor_dimensions[::-1]
			max_sensor_index = max_sensor_index[::-1]
			cal_image_dimensions = cal_image_dimensions[::-1]
			max_image_index = max_image_index[::-1]
			ref_df['px_x'], ref_df['px_y'] = ref_df['px_y'], ref_df['px_x']
			ref_df['alt'], ref_df['az'] = ref_df['az'], ref_df['alt']

		# create the px to azalt models
		from sklearn.preprocessing import PolynomialFeatures
		from sklearn.linear_model import LinearRegression

		poly1 = PolynomialFeatures(degree=3, include_bias=False)
		poly2 = PolynomialFeatures(degree=3, include_bias=False)

		ref_df.to_csv('slay ref df.csv')

		independent_poly_px = pd.DataFrame(data=poly1.fit_transform(ref_df[['px_x', 'px_y']]), columns=poly1.get_feature_names_out(ref_df[['px_x', 'px_y']].columns))
		azalt_model = LinearRegression(fit_intercept=False)
		azalt_model.fit(independent_poly_px, ref_df[['az', 'alt']])

		independent_poly_azalt = pd.DataFrame(data=poly2.fit_transform(ref_df[['az', 'alt']]), columns=poly2.get_feature_names_out(ref_df[['az', 'alt']].columns))
		px_model = LinearRegression(fit_intercept=False)
		px_model.fit(independent_poly_azalt, ref_df[['px_x', 'px_y']])
		
		self.camera_name = camera_name
		self.lens_name = lens_name
		self.zoom_factor = zoom_factor
		self.settings_notes = settings_notes
		self.sensor_dimensions = sensor_dimensions
		self.max_sensor_index = max_sensor_index
		self.cal_image_dimensions = cal_image_dimensions
		self.max_image_index = max_image_index
		self.center_px = center_px
		self.ref_df = ref_df
		self.poly1 = poly1
		self.poly2 = poly2
		self.azalt_model = azalt_model
		self.px_model = px_model
		self.pixel_map_type = pixel_map_type

		px_LT = [0-center_px[0], center_px[1]]
		px_RT = [center_px[0], center_px[1]]
		px_LB = [0-center_px[0], 0-center_px[1]]
		px_RB = [center_px[0], 0-center_px[1]]
		self.px_corners = np.concatenate((px_LT, px_RT, px_LB, px_RB)).reshape(-1,2)
		azalt_LT, azalt_RT, azalt_LB, azalt_RB = self.px_azalt_models_convert(input=self.px_corners, direction='px_to_azalt')
		self.azalt_corners = np.concatenate((azalt_LT, azalt_RT, azalt_LB, azalt_RB)).reshape(-1,2)
		px_top = [0, center_px[1]]
		px_right = [center_px[0], 0]
		px_bottom = [0, 0-center_px[1]]
		px_left = [0-center_px[0], 0]
		self.px_edges = np.concatenate((px_top, px_right, px_bottom, px_left)).reshape(-1,2)
		azalt_top, azalt_right, azalt_bottom, azalt_left = self.px_azalt_models_convert(input=self.px_edges, direction='px_to_azalt')
		self.azalt_edges = np.concatenate((azalt_top, azalt_right, azalt_bottom, azalt_left)).reshape(-1,2)


	def represent_camera(self):
		print('\nCamera Name: ', self.camera_name)
		print('Lens: ', self.lens_name)
		print('Zoom: ', self.zoom_factor)
		print('Settings Notes: ', self.settings_notes)
		print('Sensor Dimensions: ', self.sensor_dimensions)
		print('Image Dimensions: ', self.cal_image_dimensions)
		print('Pixel Corners LT, RT, LB, RB:\n', self.px_corners)
		print('Az / Alt Corners LT, RT, LB, RB:\n', self.azalt_corners)
		print('Pixel Edges Top, Right, Bottom, Left:\n', self.px_edges)
		print('Az / Alt Edges Top, Right, Bottom, Left:\n', self.azalt_edges)
		print('Pixels per degree horizontal: ', self.cal_image_dimensions[0] / (2*self.azalt_edges[1,0]))
		print('Pixels per degree vertical: ', self.cal_image_dimensions[1] / (2*self.azalt_edges[0,1]))

	# generate awim data in form of a single dictionary for embedding in any image file
	def awim_metadata_generate(self, current_image, date_gregorian_ns_time_utc, earth_latlng, center_ref, azalt_ref, img_orientation):
		if img_orientation == 'portrait':
			cam_sensor_dimensions = [self.sensor_dimensions[1], self.sensor_dimensions[0]]
		image_dimensions = current_image.size
		max_image_index = np.subtract(image_dimensions, 1)
		img_center = np.divide(max_image_index, 2)
		img_aspect_ratio = image_dimensions[0] / image_dimensions[1]
		cam_aspect_ratio = cam_sensor_dimensions[0] / cam_sensor_dimensions[1]
		if img_aspect_ratio != cam_aspect_ratio:
			print('error: image aspect ratio does not match camera aspect ratio, but it should')
		else:
			img_resize_factor = image_dimensions[0] / cam_sensor_dimensions[0]
		if center_ref == 'center':
			center_ref = img_center

		if img_orientation == 'landscape':
			azalt_model_coefficients = pd.DataFrame(self.azalt_model.coef_, columns=self.azalt_model.feature_names_in_, index=['az_predict', 'alt_predict'])
			px_model_coefficients = pd.DataFrame(self.px_model.coef_, columns=self.px_model.feature_names_in_, index=['x_px_predict', 'y_px_predict'])
		elif img_orientation == 'portrait': # TODO: make this the transpose of landscape
			azalt_model_coefficients = pd.DataFrame(self.azalt_model.coef_, columns=self.azalt_model.feature_names_in_, index=['az_predict', 'alt_predict'])
			px_model_coefficients = pd.DataFrame(self.px_model.coef_, columns=self.px_model.feature_names_in_, index=['x_px_predict', 'y_px_predict'])
		azalt_model_coefficients /= img_resize_factor
		px_model_coefficients *= img_resize_factor

		px_LT = [0-img_center[0], img_center[1]]
		px_RT = [img_center[0], img_center[1]]
		px_LB = [0-img_center[0], 0-img_center[1]]
		px_RB = [img_center[0], 0-img_center[1]]
		img_px_corners = np.concatenate((px_LT, px_RT, px_LB, px_RB)).reshape(-1,2)
		img_azalt_LT, img_azalt_RT, img_azalt_LB, img_azalt_RB = self.px_azalt_models_convert(input=np.divide(img_px_corners, img_resize_factor), direction='px_to_azalt')
		img_azalt_corners = np.concatenate((img_azalt_LT, img_azalt_RT, img_azalt_LB, img_azalt_RB)).reshape(-1,2)
		img_azalt_corners[:,0] = (img_azalt_corners[:,0] + azalt_ref[0]) % 360
		img_azalt_corners[:,1] = img_azalt_corners[:,1] + azalt_ref[1]
		px_top = [0, img_center[1]]
		px_right = [img_center[0], 0]
		px_bottom = [0, 0-img_center[1]]
		px_left = [0-img_center[0], 0]
		img_px_edges = np.concatenate((px_top, px_right, px_bottom, px_left)).reshape(-1,2)
		img_azalt_top, img_azalt_right, img_azalt_bottom, img_azalt_left = self.px_azalt_models_convert(input=np.divide(img_px_edges, img_resize_factor), direction='px_to_azalt')
		img_azalt_edges = np.concatenate((img_azalt_top, img_azalt_right, img_azalt_bottom, img_azalt_left)).reshape(-1,2)
		img_azalt_edges[:,0] = (img_azalt_edges[:,0] + azalt_ref[0]) % 360
		img_azalt_edges[:,1] = img_azalt_edges[:,1] + azalt_ref[1]

		earth_latlng_string = ', '.join(str(i) for i in earth_latlng)
		image_dimensions_string = ', '.join(str(i) for i in image_dimensions)
		center_ref_string = ', '.join(str(i) for i in center_ref)
		azalt_ref_string = ', '.join(str(i) for i in azalt_ref)
		azalt_model_coefficients_csv = azalt_model_coefficients.to_csv()
		px_model_coefficients_csv = px_model_coefficients.to_csv()
		px_corners_string = ', '.join(str(i) for i in img_px_corners)
		azalt_corners_string = ', '.join(str(i) for i in img_azalt_corners)
		px_edges_string = ', '.join(str(i) for i in img_px_edges)
		azalt_edges_string = ', '.join(str(i) for i in img_azalt_edges)

		awim_dictionary = {'Earth Lat / Long': earth_latlng_string, 'Date / Time, Gregorian NS / UTC': date_gregorian_ns_time_utc, 'Image Dimensions': image_dimensions_string, 'Center Reference Pixel': center_ref_string, 'Az / Alt Reference': azalt_ref_string, 'Az / Alt Model': azalt_model_coefficients_csv, 'Pixel Corners': px_corners_string, 'Az / Alt Corners': azalt_corners_string, 'Pixel Edges': px_edges_string, 'Az / Alt Edges': azalt_edges_string, 'Pixel Model': px_model_coefficients_csv, 'Pixel Map Type': self.pixel_map_type}

		return awim_dictionary

	def azalt_ref_from_celestial(self, image, capture_moment, earth_latlng, center_ref, what_object, object_px, img_orientation):
		image_dimensions = image.size
		max_image_index = np.subtract(image_dimensions, 1)
		img_center = np.divide(max_image_index, 2)
		if center_ref == 'center':
			center_ref = img_center
		img_aspect_ratio = image_dimensions[0] / image_dimensions[1]
		cam_aspect_ratio = self.cal_image_dimensions[0] / self.cal_image_dimensions[1]
		if img_aspect_ratio != cam_aspect_ratio:
			print('error: image aspect ratio does not match camera aspect ratio, but it should')
		else:
			img_resize_factor = image_dimensions[0] / self.sensor_dimensions[0]

		object_px_rel = [object_px[0] - center_ref[0], center_ref[1] - object_px[1]]

		img_astropy_time = Time(capture_moment)
		img_astropy_location = EarthLocation(lat=earth_latlng[0]*u.deg, lon=earth_latlng[1]*u.deg)
		img_astropy_frame = AltAz(obstime=img_astropy_time, location=img_astropy_location)
		if what_object == 'sun':
			sun_altaz = get_sun(img_astropy_time).transform_to(img_astropy_frame)

		object_azalt_rel = self.px_azalt_models_convert(input=np.divide(object_px_rel, img_resize_factor), direction='px_to_azalt')
		img_azalt_corners[:,0] = (img_azalt_corners[:,0] + azalt_ref[0]) % 360
		img_azalt_corners[:,1] = img_azalt_corners[:,1] + azalt_ref[1]


		return awim_dictionary

	def px_azalt_models_convert(self, input, direction):
		if isinstance(input, list): # models require numpy arrays
			input = np.array(input, dtype=float).reshape(-1,2)

		full_shape = input.shape # save the shape for reshape later
		output = np.zeros(full_shape)
		output_sign = np.where(input < 0, -1, 1) # models are positive values only. Save sign.
		input = np.abs(input)
		input = input.reshape(-1,2)
		if direction == 'px_to_azalt':
			input_poly = self.poly1.fit_transform(input)
			output = self.azalt_model.predict(input_poly).reshape(full_shape)
		elif direction == 'azalt_to_px':
			input_poly = self.poly2.fit_transform(input)
			output = self.px_model.predict(input_poly).reshape(full_shape)
		output = np.multiply(output, output_sign)

		return output