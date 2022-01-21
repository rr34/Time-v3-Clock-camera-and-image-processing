import math
import numpy as np

# AltAz to polar coordinates, imagining the image 2D. theta from positive altitude axis in direction of positive azimuth axis. r is "flat degrees" from center
def altaz_to_special_polar(alt, az):
	r = math.sqrt(alt**2 + az**2)
	print(alt, az)
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

# from four edge pixel coordinates of a circular target image in a calibration grid, known target angular size, and known aim pixel coordinate, calculate the hit pixel and the miss angle as relative Alt Az from the camera
def target_miss(target_top, target_right, target_bottom, target_left, target_diameter_degrees, aim_px):
	# find center of the target in the calibration image using crazy matrix math from StackOverflow = "the pixel hit by the target", kinda backwards from target hit by pixel, but...
	right_rel = np.subtract(target_right,target_left)
	bottom_rel = np.subtract(target_bottom,target_top)
	LR_scalar = np.cross(np.subtract(target_top,target_left), bottom_rel) / np.cross(right_rel, bottom_rel)
	TB_scalar = np.cross(np.subtract(target_left,target_top), right_rel) / np.cross(bottom_rel, right_rel)
	if np.cross(right_rel, bottom_rel) == 0 and np.cross(np.subtract(target_top,target_left), right_rel) == 0:
		return 'error, collinear lines given'
	if np.cross(right_rel, bottom_rel) == 0 and np.cross(np.subtract(target_top,target_left), right_rel) != 0:
		return 'error, parallel non-intersecting lines given'
	if np.cross(right_rel, bottom_rel) != 0 and LR_scalar >= 0 and LR_scalar <= 1 and TB_scalar >= 0 and TB_scalar <= 1:
		hit_px = np.add(target_left, LR_scalar*right_rel)
		hit_px2 = np.add(target_top, TB_scalar*bottom_rel) # would be the exact same as the other calculation
		if all(hit_px) != all(hit_px2):
			return 'error, something went wrong, dont know what, look here'
	# found center that was hit, now find angle between hit and aim.
	target_diameter_x_pixels = np.sqrt(np.subtract(target_right,target_left)[0]**2+np.subtract(target_right,target_left)[1]**2)
	target_diameter_y_pixels = np.sqrt(np.subtract(target_bottom,target_top)[0]**2+np.subtract(target_bottom,target_top)[1]**2)
	pixel_delta_x = target_diameter_degrees / target_diameter_x_pixels # degrees per pixel horizontal
	pixel_delta_y = target_diameter_degrees / target_diameter_y_pixels # degrees per pixel vertical
	target_pos_alt_rel = (hit_px[1] - aim_px[1]) * pixel_delta_y # camera aim error vertical degrees
	target_pos_az_rel = (hit_px[0] - aim_px[0]) * pixel_delta_x # camera aim error horizontal degrees
	target_pos_altaz_rel = [target_pos_alt_rel, target_pos_az_rel]
	return hit_px, target_pos_altaz_rel

# from two edge pixel coordinates of known angular size alignment targets, angular distance of the targets from grid center, and the target hit pixel of the grid, calculate the angular rotation error of the entire grid, CCW is positive.
def grid_rotation_error(grid_align1_px, grid_align2_px, orientation, grid_align_altaz_reltarget, targets_diameter_degrees, hit_px):
	# find error rotation angle using vertical grid align target
	if orientation == 'horizontal':
		xy_index = 1 # align with the y pixel coordinate
		altaz_index = 1
	elif orientation == 'vertical':
		xy_index = 0 # align with the x pixel coordinate
		altaz_index = 0
	else:
		print('must specify orientation')

	grid_align_center_px = np.divide(np.add(grid_align_v_left, grid_align_v_right), 2)
	grid_align_diameter_px = np.sqrt((grid_align2_px[0] - grid_align1_px[0])**2 + (grid_align2_px[1] - grid_align1_px[1])**2)
	pixel_delta = targets_diameter_degrees / grid_align_diameter_px # degrees per pixel at the alignment point
	miss = (hit_px[xy_index] - grid_align_center_px[xy_index]) * pixel_delta # grid align pixel not aligned with hit target pixel in degrees at grid align point
	grid_rotation_error_degreesCCW = math.atan(miss / grid_align_altaz_reltarget[altaz_index]) * 180/math.pi # converted to a rotation angle
	rotation_error_h = math.atan(miss_h / horizontal_align_degrees_rel[1]) * 180/math.pi
	print(rotation_error)
	return (rotation_error_v + rotation_error_h) / 2
