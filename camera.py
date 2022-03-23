
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
		camera_name = cal_df[cal_df['type'] == 'camera_name']['misc'].iat[0]
		lens_name = cal_df[cal_df['type'] == 'lens_name']['misc'].iat[0]
		zoom_factor = cal_df[cal_df['type'] == 'zoom_factor']['misc'].iat[0]
		settings_notes = cal_df[cal_df['type'] == 'settings_notes']['misc'].iat[0]
		cam_image_dimensions = [int(cal_df[cal_df['type'] == 'cam_image_dimensions']['x_rec'].iat[0]), int(cal_df[cal_df['type'] == 'cam_image_dimensions']['y_rec'].iat[0])]
		max_image_index = np.subtract(cam_image_dimensions, 1)
		pixel_map_type = cal_df[cal_df['type'] == 'pixel_map_type']['misc'].iat[0]
		center_px = max_image_index / 2 # usually x.5, non-integer, bc even number of pixels means center is a boundary rather than a pixel
		calibration_orientation = cal_df[cal_df['type'] == 'orientation']['misc'].iat[0]
		targets_diameter_degrees = float(cal_df[cal_df['type'] == 'targets_diameter']['misc'].iat[0])
		calibration_quadrant = cal_df[cal_df['type'] == 'quadrant']['misc'].iat[0]


		if calibration_orientation == 'portrait':
			center_px = center_px[::-1]
			cam_image_dimensions = cam_image_dimensions[::-1]
			max_image_index = max_image_index[::-1]



		# first for loop: each calc step represented by a "calculate_and_record" row: get vars, check if there, handle cases
		# crosshairs case 1: record center as original crosshairs
		# crosshairs case 2. on xang axis (vertical)
		# crosshairs case 3. on yang axis (horizontal)
		# crosshairs case 4. not on an axis, "floating"
		# target miss: calculate the target position xyangs relative to the original crosshairs, used for all ref points with same xang,yang _sect
		# NOT USED, calculate the target center from edges just to compare to the actual recorded center, which is used
		# rotation error: (°CCW) from the align points, used for all ref points with same xang,yang _sect
		# pixel delta: °/px used for adjusting pixel location (instead of xyangs) for "floating" crosshairs
		# IF NOT BLANK marks the "check for values" if statement
		for row in cal_df[cal_df['type'] == 'calculate_and_record'].iterrows():
			# prepare variables
			xyangs_sect = row[1][['xang_sect', 'yang_sect']].values
			xyangs_rel = row[1][['xang_rel', 'yang_rel']].values
			xyangs_total = [xyangs_sect[0]+xyangs_rel[0], xyangs_sect[1]+xyangs_rel[1]]
			cal_df.loc[row[0], 'xang_total'] = xyangs_total[0]
			cal_df.loc[row[0], 'yang_total'] = xyangs_total[1]

			df_find_rows_section = (cal_df['xang_sect']==row[1]['xang_sect']) & (cal_df['yang_sect']==row[1]['yang_sect'])
			df_section = cal_df[df_find_rows_section]
			df_find_rows_pt = (cal_df['xang_sect']==row[1]['xang_sect']) & (cal_df['yang_sect']==row[1]['yang_sect']) & (cal_df['xang_rel']==row[1]['xang_rel']) & (cal_df['yang_rel']==row[1]['yang_rel'])
			df_pt = cal_df[df_find_rows_pt]

			df_find_row_target = (df_section['misc']=='target_center')
			target_center_section = df_section[df_find_row_target][['x_rec', 'y_rec']].values[0]

			df_find_crosshairs_sect = (cal_df['xang_total']==xyangs_sect[0]) & (cal_df['yang_total']==xyangs_sect[1]) & (cal_df['type']=='calculate_and_record') & (cal_df['misc']=='this_section_crosshairs')
			crosshairs_section = cal_df[df_find_crosshairs_sect][['out1', 'out2']].values[0]

			if row[1]['misc'] == 'this_section_crosshairs':
				if xyangs_total[0]==0 and xyangs_total[1]==0: # origin only
					cal_df.loc[row[0], 'out1'] = center_px[0]
					cal_df.loc[row[0], 'out2'] = center_px[1]
				else:
					# retrieve crosshairs from previous section(s)
					df_find_crosshairs = (cal_df['xang_total']==xyangs_total[0]) & (cal_df['yang_total']==xyangs_total[1]) & (cal_df['type']=='calculate_and_record') & (cal_df['misc']=='crosshairs')
					crosshairs_array = cal_df[df_find_crosshairs][['out1', 'out2']]
					not_blank, crosshairs = _coord_standard(crosshairs_array)
					if not_blank: # IF NOT BLANK
						if isinstance(crosshairs, np.ndarray):
							crosshairs1 = crosshairs[0,:]
							crosshairs2 = crosshairs[1,:]
							crosshairs = [(crosshairs1[0]+crosshairs2[0])/2, (crosshairs1[1]+crosshairs2[1])/2]
						cal_df.loc[row[0], 'out1'] = crosshairs[0]
						cal_df.loc[row[0], 'out2'] = crosshairs[1]

			elif row[1]['misc'] == 'crosshairs':
				df_find_recorded = (df_pt['misc']=='center') & (df_pt['type']=='ref_point')
				px = df_pt[df_find_recorded][['x_rec', 'y_rec']]
				not_blank, px = _coord_standard(px)
				# if a px is recorded, case 2 on xang axis, case 3 on yang axis, case 4 on neither axis
				if not_blank: # IF NOT BLANK
					if xyangs_total[0]==0:
						px_dist = math.sqrt((px[0] - target_center_section[0])**2 + (px[1] - target_center_section[1])**2)
						crosshairs_px = [center_px[0], crosshairs_section[1] + px_dist]
						cal_df.loc[row[0], 'out1'] = crosshairs_px[0]
						cal_df.loc[row[0], 'out2'] = crosshairs_px[1]
					elif xyangs_total[1]==0:
						px_dist = math.sqrt((px[0] - target_center_section[0])**2 + (px[1] - target_center_section[1])**2)
						crosshairs_px = [crosshairs_section[0] - px_dist, center_px[1]]
						cal_df.loc[row[0], 'out1'] = crosshairs_px[0]
						cal_df.loc[row[0], 'out2'] = crosshairs_px[1]
					else: # adjust the pixel for crosshairs, not the xyangs
						df_find_row_tpaara = (df_section['misc']=='target_pos_xyangs_relaim')
						target_pos_xyangs_relaim = df_section[df_find_row_tpaara][['out1', 'out2']].values[0]
						df_find_row_pdaa_v = (df_section['misc']=='rotation_error_v_and_px_delta')
						pixel_delta_at_align_v = df_section[df_find_row_pdaa_v][['out2']].values[0]
						df_find_row_pdaa_h = (df_section['misc']=='rotation_error_h_and_px_delta')
						pixel_delta_at_align_h = df_section[df_find_row_pdaa_h][['out2']].values[0]
						# note there is no grid rotation error for these cases because not on an axis
						crosshairs_px = np.array([1,2])
						crosshairs_px[0] = px[0] + pixel_delta_at_align * target_pos_xyangs_relaim[0]
						crosshairs_px[1] = px[1] - pixel_delta_at_align * target_pos_xyangs_relaim[1]
						cal_df.loc[row[0], 'out1'] = crosshairs_px[0]
						cal_df.loc[row[0], 'out2'] = crosshairs_px[1]

			elif row[1]['misc'] == 'target_pos_xyangs_relaim':
				target_top = df_pt[df_pt['misc']=='target_top'][['x_rec', 'y_rec']].values[0]
				target_right = df_pt[df_pt['misc']=='target_right'][['x_rec', 'y_rec']].values[0]
				target_bottom = df_pt[df_pt['misc']=='target_bottom'][['x_rec', 'y_rec']].values[0]
				target_left = df_pt[df_pt['misc']=='target_left'][['x_rec', 'y_rec']].values[0]

				target_pos_xyangs_relaim = _target_miss(target_left, target_right, target_top, target_bottom, target_center_section, targets_diameter_degrees, crosshairs_section)
				cal_df.loc[row[0], 'out1'] = target_pos_xyangs_relaim[0]
				cal_df.loc[row[0], 'out2'] = target_pos_xyangs_relaim[1]

			elif row[1]['misc'] == 'target_center_calculated_from_edges':
				target_top = df_pt[df_pt['misc']=='target_top'][['x_rec', 'y_rec']].values[0]
				target_right = df_pt[df_pt['misc']=='target_right'][['x_rec', 'y_rec']].values[0]
				target_bottom = df_pt[df_pt['misc']=='target_bottom'][['x_rec', 'y_rec']].values[0]
				target_left = df_pt[df_pt['misc']=='target_left'][['x_rec', 'y_rec']].values[0]
				if not math.isnan(target_top[0]): # IF NOT BLANK
					target_pos_calculated_notused = _intersect_two_lines(target_left, target_right, target_top, target_bottom)
					cal_df.loc[row[0], 'out1'] = target_pos_calculated_notused[0]
					cal_df.loc[row[0], 'out2'] = target_pos_calculated_notused[1]

			elif row[1]['misc'] == 'rotation_error_v_and_px_delta':
				align_orientation = 'vertical'
				align1_px = df_section[df_section['misc']=='vertical_left'][['x_rec', 'y_rec']].values[0]
				align2_px = df_section[df_section['misc']=='vertical_right'][['x_rec', 'y_rec']].values[0]
				if not math.isnan(align1_px[0]): # IF NOT BLANK
					grid_rotation_error_degreesCCW, pixel_delta_at_align = _grid_rotation_error(align_orientation=align_orientation, align1_px=align1_px, align2_px=align2_px, align_xyangs_reltarget=xyangs_rel, targets_diameter_degrees=targets_diameter_degrees, target_pos_px=target_center_section)
					cal_df.loc[row[0], 'out1'] = grid_rotation_error_degreesCCW
					cal_df.loc[row[0], 'out2'] = pixel_delta_at_align

			elif row[1]['misc'] == 'rotation_error_h_and_px_delta':
				align_orientation = 'horizontal'
				align1_px = df_section[df_section['misc']=='horizontal_top'][['x_rec', 'y_rec']].values[0]
				align2_px = df_section[df_section['misc']=='horizontal_bottom'][['x_rec', 'y_rec']].values[0]
				if not math.isnan(align1_px[0]): # IF NOT BLANK
					grid_rotation_error_degreesCCW, pixel_delta_at_align = _grid_rotation_error(align_orientation=align_orientation, align1_px=align1_px, align2_px=align2_px, align_xyangs_reltarget=xyangs_rel, targets_diameter_degrees=targets_diameter_degrees, target_pos_px=target_center_section)
					cal_df.loc[row[0], 'out1'] = grid_rotation_error_degreesCCW
					cal_df.loc[row[0], 'out2'] = pixel_delta_at_align



		# second for loop: each reference point represented by a "ref_point" row
		# ref point case 1: record the center as a reference point bc defining as original xyangs = [0,0]
		# ref point case 2: like crosshairs case 2, on xang axis (vertical)
		# ref point case 3: like crosshairs case 3. on yang axis (horizontal)
		# ref point case 4: normal ref point
		for row in cal_df[cal_df['type'] == 'ref_point'].iterrows():
			# prepare variables
			px = row[1][['x_rec', 'y_rec']].values
			xyangs_sect = row[1][['xang_sect', 'yang_sect']].values
			xyangs_rel = row[1][['xang_rel', 'yang_rel']].values
			xyangs_total = [xyangs_sect[0]+xyangs_rel[0], xyangs_sect[1]+xyangs_rel[1]]
			cal_df.loc[row[0], 'xang_total'] = xyangs_total[0]
			cal_df.loc[row[0], 'yang_total'] = xyangs_total[1]

			df_find_rows_section = (cal_df['xang_sect']==row[1]['xang_sect']) & (cal_df['yang_sect']==row[1]['yang_sect'])
			df_section = cal_df[df_find_rows_section]
			df_find_rows_pt = (cal_df['xang_sect']==row[1]['xang_sect']) & (cal_df['yang_sect']==row[1]['yang_sect']) & (cal_df['xang_rel']==row[1]['xang_rel']) & (cal_df['yang_rel']==row[1]['yang_rel'])
			df_pt = cal_df[df_find_rows_pt]

			df_find_row_target = (df_section['misc']=='target_center')
			target_center_section = df_section[df_find_row_target][['x_rec', 'y_rec']].values[0]

			df_find_crosshairs_sect = (cal_df['xang_total']==xyangs_sect[0]) & (cal_df['yang_total']==xyangs_sect[1]) & (cal_df['type']=='calculate_and_record') & (cal_df['misc']=='this_section_crosshairs')
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
				if xyangs_sect[0]==0 and xyangs_sect[1]==0 and not math.isnan(grid_rotation_error_degreesCCW_v) and not math.isnan(grid_rotation_error_degreesCCW_h):
					grid_rotation_error_degreesCCW = (grid_rotation_error_degreesCCW_v + grid_rotation_error_degreesCCW_h) / 2
				elif xyangs_sect[0]==0 and xyangs_sect[1]!=0 and not math.isnan(grid_rotation_error_degreesCCW_v):
					grid_rotation_error_degreesCCW = grid_rotation_error_degreesCCW_v
				elif xyangs_sect[0]!=0 and xyangs_sect[1]==0 and not math.isnan(grid_rotation_error_degreesCCW_h):
					grid_rotation_error_degreesCCW = grid_rotation_error_degreesCCW_h
				else:
					grid_rotation_error_degreesCCW = 0 # CATCH-ALL FOR BLANKS
			
				# record the ref pixels for the 4 cases
				if xyangs_total[0]==0 and xyangs_total[1]==0: # origin
					cal_df.loc[row[0], 'x_px'] = center_px[0]
					cal_df.loc[row[0], 'y_px'] = center_px[1]
					cal_df.loc[row[0], 'xang'] = 0
					cal_df.loc[row[0], 'yang'] = 0
				elif xyangs_total[0]==0 and xyangs_total[1]!=0: # on the y axis
					px_dist = math.sqrt((px[0] - target_center_section[0])**2 + (px[1] - target_center_section[1])**2)
					cal_df.loc[row[0], 'x_px'] = center_px[0]
					cal_df.loc[row[0], 'y_px'] = crosshairs_section[1] + px_dist
					cal_df.loc[row[0], 'xang'] = xyangs_total[0]
					cal_df.loc[row[0], 'yang'] = xyangs_total[1]
				elif xyangs_total[0]!=0 and xyangs_total[1]==0: # on the x axis
					px_dist = math.sqrt((px[0] - target_center_section[0])**2 + (px[1] - target_center_section[1])**2)
					cal_df.loc[row[0], 'x_px'] = crosshairs_section[0] - px_dist
					cal_df.loc[row[0], 'y_px'] = center_px[1]
					cal_df.loc[row[0], 'xang'] = xyangs_total[0]
					cal_df.loc[row[0], 'yang'] = xyangs_total[1]
				else: # anywhere else
					target_pos_xyangs_relaim = df_section[(df_section['misc']=='target_pos_xyangs_relaim')][['out1', 'out2']].values[0]
					xyangs_polar = _xyangs_to_special_polar(xyangs_rel)
					grid_rotation_error_xyangs_relgridpoint =  [math.tan(grid_rotation_error_degreesCCW*math.pi/180) * xyangs_polar[0] * -1*math.sin(xyangs_polar[1]*math.pi/180), math.tan(grid_rotation_error_degreesCCW*math.pi/180) * xyangs_polar[0] * math.cos(xyangs_polar[1]*math.pi/180)]
					cal_df.loc[row[0], 'x_px'] = px[0]
					cal_df.loc[row[0], 'y_px'] = px[1]
					cal_df.loc[row[0], 'xang'] = xyangs_total[0] + target_pos_xyangs_relaim[0] + grid_rotation_error_xyangs_relgridpoint[0]
					cal_df.loc[row[0], 'yang'] = xyangs_total[1] + target_pos_xyangs_relaim[1] + grid_rotation_error_xyangs_relgridpoint[1]

		cal_df.to_csv('slay cal output.csv')
		ref_df_filter = pd.notnull(cal_df['x_px'])
		ref_df = pd.DataFrame(cal_df[ref_df_filter][['x_px', 'y_px', 'xang', 'yang']])

		# standardize the reference pixels:
		# center, convert to positive values (RB quadrant in PhotoShop coordinate system), then orient landscape
		ref_df['x_px'] = ref_df['x_px'] - center_px[0]
		ref_df['y_px'] = ref_df['y_px'] - center_px[1]
		if calibration_quadrant == 'LB':
			ref_df['x_px'] = ref_df['x_px'] * -1
			ref_df['xang'] = ref_df['xang'] * -1
			ref_df['yang'] = ref_df['yang'] * -1
		if calibration_orientation == 'portrait':
			center_px = center_px[::-1]
			cam_image_dimensions = cam_image_dimensions[::-1]
			max_image_index = max_image_index[::-1]
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
		
		self.camera_name = camera_name
		self.lens_name = lens_name
		self.zoom_factor = zoom_factor
		self.settings_notes = settings_notes
		self.cam_image_dimensions = cam_image_dimensions
		self.max_image_index = max_image_index
		self.center_px = center_px
		self.ref_df = ref_df
		self.poly1 = poly1
		self.poly2 = poly2
		self.xyangs_model = xyangs_model
		self.px_model = px_model
		self.pixel_map_type = pixel_map_type

		px_LT = [0-center_px[0], center_px[1]]
		px_RT = [center_px[0], center_px[1]]
		px_LB = [0-center_px[0], 0-center_px[1]]
		px_RB = [center_px[0], 0-center_px[1]]
		self.px_corners = np.concatenate((px_LT, px_RT, px_LB, px_RB)).reshape(-1,2)
		xyangs_LT, xyangs_RT, xyangs_LB, xyangs_RB = self.px_xyangs_models_convert(input=self.px_corners, direction='px_to_xyangs')
		self.xyangs_corners = np.concatenate((xyangs_LT, xyangs_RT, xyangs_LB, xyangs_RB)).reshape(-1,2)
		px_top = [0, center_px[1]]
		px_right = [center_px[0], 0]
		px_bottom = [0, 0-center_px[1]]
		px_left = [0-center_px[0], 0]
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
		if img_orientation == 'landscape':
			cam_image_dimensions = self.cam_image_dimensions
		elif img_orientation == 'portrait':
			cam_image_dimensions = [self.cam_image_dimensions[1], self.cam_image_dimensions[0]]
		img_aspect_ratio = img_dimensions[0] / img_dimensions[1]
		cam_aspect_ratio = cam_image_dimensions[0] / cam_image_dimensions[1]
		if img_aspect_ratio != cam_aspect_ratio:
			print('error: image aspect ratio does not match camera aspect ratio, but it should')
		else:
			img_resize_factor = img_dimensions[0] / cam_image_dimensions[0]
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
# TODO fix this. It's pretty close, but not quite right (too simple?)
# xyangs to polar coordinates, imagining the image 2D planar.
# theta is CCW from the positive x axis. r is degrees from center
def _xyangs_to_special_polar(xyangs):
	xang_rad = xyangs[0] * math.pi/180
	yang_rad = xyangs[1] * math.pi/180
	xyangs_hyp_rad = math.sqrt(xang_rad**2 + yang_rad**2)

	if xang_rad == 0 and yang_rad == 0:
		theta_rad = 0
	elif xang_rad >= 0 and yang_rad == 0:
		theta_rad = 0
	elif xang_rad == 0 and yang_rad > 0:
		theta_rad = math.pi/2
	elif xang_rad < 0 and yang_rad == 0:
		theta_rad = math.pi
	elif xang_rad == 0 and yang_rad < 0:
		theta_rad = 3/2*math.pi
	elif xang_rad > 0 and yang_rad > 0:
		theta_rad = math.atan(yang_rad / xang_rad)
	elif xang_rad < 0 and yang_rad != 0:
		theta_rad = math.atan(yang_rad / xang_rad) + math.pi
	elif xang_rad > 0 and yang_rad < 0:
		theta_rad = (math.atan(yang_rad / xang_rad) + 2*math.pi) % (2*math.pi)

	theta = theta_rad * 180/math.pi
	xyangs_hyp = xyangs_hyp_rad * 180/math.pi

	return  [xyangs_hyp, theta]

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

# from four edge pixel coordinates of a circular target image in a calibration grid, known target angular size, and known aim pixel coordinate, calculate the hit pixel and the miss angle as relative xang,yang
def _target_miss(target_left, target_right, target_top, target_bottom, target_pos_px, targets_diameter_degrees, crosshairs):
	target_diameter_x_pixels = np.sqrt((target_right[0]-target_left[0])**2 + (target_right[1]-target_left[1])**2)
	target_diameter_y_pixels = np.sqrt((target_bottom[0]-target_top[0])**2 + (target_bottom[1]-target_top[1])**2)
	pixel_delta_x = targets_diameter_degrees / target_diameter_x_pixels # degrees per pixel horizontal
	pixel_delta_y = targets_diameter_degrees / target_diameter_y_pixels # degrees per pixel vertical
	target_pos_xang_rel = (target_pos_px[0] - crosshairs[0]) * pixel_delta_x # camera aim error xang
	target_pos_yang_rel = -1 * (target_pos_px[1] - crosshairs[1]) * pixel_delta_y # camera aim error yang, note (-)y_px is (+)yang!
	target_pos_xyangs_rel = [target_pos_xang_rel, target_pos_yang_rel]
	return target_pos_xyangs_rel

# from two edge pixel coordinates of known-angular-size grid alignment point, angular distance of the alignment point from grid target, and the target position in the image, calculate the angular rotation error of the entire grid,
# CCW grid rotation in photo is positive. NOTE: This is the opposite of the convention I use later for image tilt where CW image tilt is positive.
def _grid_rotation_error(align_orientation, align1_px, align2_px, align_xyangs_reltarget, targets_diameter_degrees, target_pos_px):
	# find error rotation angle using vertical grid align target
	if align_orientation == 'horizontal':
		xy_index = 1 # align with the y pixel coordinate
		xyangs_index = 0 # use the xang for the distance from center in degrees
	elif align_orientation == 'vertical':
		xy_index = 0 # align with the x pixel coordinate
		xyangs_index = 1 # use the yang for the distance from center in degrees
	else:
		print('must specify align_orientation')

	align_center_px = np.divide(np.add(align1_px, align2_px), 2)
	align_diameter_pxs = np.sqrt((align2_px[0] - align1_px[0])**2 + (align2_px[1] - align1_px[1])**2)
	pixel_delta = targets_diameter_degrees / align_diameter_pxs # degrees per pixel at the alignment point
	miss_ang_rel = (align_center_px[xy_index] - target_pos_px[xy_index]) * pixel_delta # grid align pixel not in same pixel line with hit target pixel in degrees at grid align point
	grid_rotation_error_degreesCCW = math.atan(miss_ang_rel / (-1*align_xyangs_reltarget[xyangs_index])) * 180/math.pi # converted to a rotation angle

	return grid_rotation_error_degreesCCW, pixel_delta

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