import math
import numpy as np
import pickle
import pandas as pd

# TODO fix this. It's pretty close, but not quite right (too simple?)
def xysph_to_special_polar(xysph):
	xsph_rad = xysph[0] * math.pi/180
	ysph_rad = xysph[1] * math.pi/180
	xysph_hyp_rad = math.sqrt(xsph_rad**2 + ysph_rad**2)

	if xsph_rad == 0 and ysph_rad == 0:
		theta_rad = 0
	elif xsph_rad >= 0 and ysph_rad == 0:
		theta_rad = 0
	elif xsph_rad == 0 and ysph_rad > 0:
		theta_rad = math.pi/2
	elif xsph_rad < 0 and ysph_rad == 0:
		theta_rad = math.pi
	elif xsph_rad == 0 and ysph_rad < 0:
		theta_rad = 3/2*math.pi
	elif xsph_rad > 0 and ysph_rad > 0:
		theta_rad = math.atan(ysph_rad / xsph_rad)
	elif xsph_rad < 0 and ysph_rad != 0:
		theta_rad = math.atan(ysph_rad / xsph_rad) + math.pi
	elif xsph_rad > 0 and ysph_rad < 0:
		theta_rad = (math.atan(ysph_rad / xsph_rad) + 2*math.pi) % (2*math.pi)

	theta = theta_rad * 180/math.pi
	xysph_hyp = xysph_hyp_rad * 180/math.pi

	return  [xysph_hyp, theta]

# xysph to polar coordinates, imagining the image 2D planar.
# theta is CCW from the positive x axis. r is degrees from center
def xysph_to_polar(xysph):

	xsph_rad = xysph[0] * math.pi/180
	ysph_rad = xysph[1] * math.pi/180
	xyhyp_rad = math.pi/2 - math.asin(math.cos(xsph_rad) * math.cos(ysph_rad))

	if xsph_rad >= 0 and ysph_rad == 0:
		theta_rad = 0
	elif xsph_rad == 0 and ysph_rad > 0:
		theta_rad = math.pi/2
	elif xsph_rad < 0 and ysph_rad == 0:
		theta_rad = math.pi
	elif xsph_rad == 0 and ysph_rad < 0:
		theta_rad = 3*math.pi/2
	elif ysph_rad > 0:
		theta_rad = math.pi/2 - math.asin(math.tan(math.pi/2-xyhyp_rad) * math.tan(xsph_rad))
	elif ysph_rad < 0:
		theta_rad = -1 * (math.pi/2 - math.asin(math.tan(math.pi/2-xyhyp_rad) * math.tan(xsph_rad))) + 2*math.pi

	xyhyp = xyhyp_rad * 180/math.pi
	theta = theta_rad * 180/math.pi

	return [xyhyp, theta]

# crazy matrix math from StackOverflow:
def intersect_two_lines(line_1_1, line_1_2, line_2_1, line_2_2):
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

# from four edge pixel coordinates of a circular target image in a calibration grid, known target angular size, and known aim pixel coordinate, calculate the hit pixel and the miss angle as relative xsph,ysph
def target_miss(target_left, target_right, target_top, target_bottom, target_pos_px, targets_diameter_degrees, crosshairs):
	target_diameter_x_pixels = np.sqrt((target_right[0]-target_left[0])**2 + (target_right[1]-target_left[1])**2)
	target_diameter_y_pixels = np.sqrt((target_bottom[0]-target_top[0])**2 + (target_bottom[1]-target_top[1])**2)
	pixel_delta_x = targets_diameter_degrees / target_diameter_x_pixels # degrees per pixel horizontal
	pixel_delta_y = targets_diameter_degrees / target_diameter_y_pixels # degrees per pixel vertical
	target_pos_xsph_rel = (target_pos_px[0] - crosshairs[0]) * pixel_delta_x # camera aim error xsph
	target_pos_ysph_rel = -1 * (target_pos_px[1] - crosshairs[1]) * pixel_delta_y # camera aim error ysph, note (-)y_px is (+)ysph!
	target_pos_xysph_rel = [target_pos_xsph_rel, target_pos_ysph_rel]
	return target_pos_xysph_rel

# from two edge pixel coordinates of known-angular-size grid alignment point, angular distance of the alignment point from grid target, and the target position in the image, calculate the angular rotation error of the entire grid,
# CCW grid rotation in photo is positive. NOTE: This is the opposite of the convention I use later for image tilt where CW image tilt is positive.
def grid_rotation_error(align_orientation, align1_px, align2_px, align_xysph_reltarget, targets_diameter_degrees, target_pos_px):
	# find error rotation angle using vertical grid align target
	if align_orientation == 'horizontal':
		xy_index = 1 # align with the y pixel coordinate
		xysph_index = 0 # use the xsph for the distance from center in degrees
	elif align_orientation == 'vertical':
		xy_index = 0 # align with the x pixel coordinate
		xysph_index = 1 # use the ysph for the distance from center in degrees
	else:
		print('must specify align_orientation')

	align_center_px = np.divide(np.add(align1_px, align2_px), 2)
	align_diameter_pxs = np.sqrt((align2_px[0] - align1_px[0])**2 + (align2_px[1] - align1_px[1])**2)
	pixel_delta = targets_diameter_degrees / align_diameter_pxs # degrees per pixel at the alignment point
	miss_sph_rel = (align_center_px[xy_index] - target_pos_px[xy_index]) * pixel_delta # grid align pixel not in same pixel line with hit target pixel in degrees at grid align point
	grid_rotation_error_degreesCCW = math.atan(miss_sph_rel / (-1*align_xysph_reltarget[xysph_index])) * 180/math.pi # converted to a rotation angle

	return grid_rotation_error_degreesCCW, pixel_delta

# simply standardize coordinate type:
# single coordinate is a list. more than one is a 2-dim Numpy array of two columns
def coord_standard(obj):
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