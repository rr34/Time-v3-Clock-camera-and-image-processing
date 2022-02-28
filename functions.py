import math
import numpy as np
import pickle

# AzAlt to polar coordinates, imagining the image 2D. theta from positive altitude axis in direction of positive azimuth axis. r is "flat degrees" from center
def azalt_to_special_polar(az, alt):
	r = math.sqrt(alt**2 + az**2)
	if alt == 0 and az == 0:
		theta = 0
	elif alt == 0 and az < 0:
		theta = 270
	elif alt == 0 and az > 0:
		theta = 90
	elif alt > 0 and az >= 0:
		theta = math.atan(az / alt) * 180/math.pi
	elif alt < 0 and az >= 0:
		theta = math.atan(az / alt) * 180/math.pi + 180
	elif alt < 0 and az <= 0:
		theta = math.atan(az / alt) * 180/math.pi + 180
	elif alt > 0 and az <= 0:
		theta = math.atan(az / alt) * 180/math.pi + 360
	return theta, r

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

# from four edge pixel coordinates of a circular target image in a calibration grid, known target angular size, and known aim pixel coordinate, calculate the hit pixel and the miss angle as relative Alt Az from the camera
def target_miss(target_left, target_right, target_top, target_bottom, target_pos_px, targets_diameter_degrees, crosshairs):
	target_diameter_x_pixels = np.sqrt((target_right[0]-target_left[0])**2 + (target_right[1]-target_left[1])**2)
	target_diameter_y_pixels = np.sqrt((target_bottom[0]-target_top[0])**2 + (target_bottom[1]-target_top[1])**2)
	pixel_delta_x = targets_diameter_degrees / target_diameter_x_pixels # degrees per pixel horizontal
	pixel_delta_y = targets_diameter_degrees / target_diameter_y_pixels # degrees per pixel vertical
	target_pos_az_rel = (target_pos_px[0] - crosshairs[0]) * pixel_delta_x # camera aim error azimuth
	target_pos_alt_rel = -1 * (target_pos_px[1] - crosshairs[1]) * pixel_delta_y # camera aim error altitude, note (-)y is (+)alt!
	target_pos_azalt_rel = [target_pos_az_rel, target_pos_alt_rel]
	return target_pos_azalt_rel

# from two edge pixel coordinates of known-angular-size grid alignment point, angular distance of the alignment point from grid target, and the target position in the image, calculate the angular rotation error of the entire grid, CCW is positive.
def grid_rotation_error(align_orientation, align1_px, align2_px, align_azalt_reltarget, targets_diameter_degrees, target_pos_px):
	# find error rotation angle using vertical grid align target
	if align_orientation == 'horizontal':
		xy_index = 1 # align with the y pixel coordinate
		azalt_index = 0 # use the azimuth for the distance from center in degrees
	elif align_orientation == 'vertical':
		xy_index = 0 # align with the x pixel coordinate
		azalt_index = 1 # use the altitude for the distance from center in degrees
	else:
		print('must specify align_orientation')

	align_center_px = np.divide(np.add(align1_px, align2_px), 2)
	align_diameter_px = np.sqrt((align2_px[0] - align1_px[0])**2 + (align2_px[1] - align1_px[1])**2)
	pixel_delta = targets_diameter_degrees / align_diameter_px # degrees per pixel at the alignment point
	miss_azalt_rel = (target_pos_px[xy_index] - align_center_px[xy_index]) * pixel_delta # grid align pixel not in same pixel line with hit target pixel in degrees at grid align point
	grid_rotation_error_degreesCCW = math.atan(miss_azalt_rel / align_azalt_reltarget[azalt_index]) * 180/math.pi # converted to a rotation angle

	return grid_rotation_error_degreesCCW, pixel_delta