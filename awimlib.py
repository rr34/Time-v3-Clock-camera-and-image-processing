import os, io, ast, re
import json
import math
import numpy as np
import PIL
import datetime
import pytz
import pandas as pd
import astropytools
import metadata_tools, formatters


def generate_empty_AWIMtag_dictionary(default_units=True):
    AWIMtag_dictionary = {}
    AWIMtag_dictionary['Location'] = [-999.9, -999.9]
    AWIMtag_dictionary['Location Unit'] = 'Latitude, Longitude; to 6 decimal places so ~11cm'
    AWIMtag_dictionary['Location Source'] = ''
    AWIMtag_dictionary['Location Altitude'] = -999.9
    AWIMtag_dictionary['Location Altitude Unit'] = 'Meters above sea level; to 1 decimal place so 10cm'
    AWIMtag_dictionary['Location Altitude Source'] = ''
    AWIMtag_dictionary['Location AGL'] = -999.9
    AWIMtag_dictionary['Location AGL Unit'] = 'Meters above ground level; to 2 decimal places so 1cm'
    AWIMtag_dictionary['Location AGL Source'] = ''
    AWIMtag_dictionary['Capture Moment'] = '0000-01-01T00:00:00Z'
    AWIMtag_dictionary['Capture Moment Unit'] = 'Gregorian New Style Calendar in ISO 8601 YYYY:MM:DDTHH:MM:SSZ'
    AWIMtag_dictionary['Capture Moment Source'] = ''
    AWIMtag_dictionary['Pixel Angle Models Type'] = '' # 3d_degree_poly_fit_abs_from_center
    AWIMtag_dictionary['Ref Pixel'] = [-999.9, -999.9]
    AWIMtag_dictionary['Ref Pixel Coord Type'] = 'top-left is (0,0) so standard; to 1 decimal so to tenth of a pixel'
    AWIMtag_dictionary['Ref Pixel Azimuth Artifae'] = [-999.9, -999.9]
    AWIMtag_dictionary['Ref Pixel Azimuth Artifae Source'] = ''
    AWIMtag_dictionary['Ref Pixel Azimuth Artifae Unit'] = 'Degrees; to hundredth of a degree'
    AWIMtag_dictionary['Grid Pixels'] = []
    AWIMtag_dictionary['Grid Angles'] = []
    AWIMtag_dictionary['Grid Azimuth Artifae'] = []
    AWIMtag_dictionary['Grid RA Dec'] = []
    AWIMtag_dictionary['RA Dec Unit'] = 'ICRS J2000 Epoch, to thousandth of an hour, hundredth of a degree'
    AWIMtag_dictionary['Pixel Size Center Horizontal Vertical'] = [-999.9, -999.9]
    AWIMtag_dictionary['Pixel Size Average Horizontal Vertical'] = [-999.9, -999.9]
    AWIMtag_dictionary['Pixel Size Unit'] = 'Pixels per Degree; to tenth of a pixel'

    if not default_units:
        AWIMtag_dictionary['Location Unit'] = ''
        AWIMtag_dictionary['Location Altitude Unit'] = ''
        AWIMtag_dictionary['Location AGL Unit'] = ''
        AWIMtag_dictionary['Capture Moment Unit'] = ''
        AWIMtag_dictionary['Ref Pixel Coord Type'] = ''
        AWIMtag_dictionary['Ref Pixel Azimuth Artifae Unit'] = ''
        AWIMtag_dictionary['RA Dec Unit'] = ''
        AWIMtag_dictionary['Pixel Size Unit'] = ''

    return AWIMtag_dictionary


def get_ref_px_and_thirds_grid_TBLR(source_image_path, ref_px):
    with PIL.Image.open(source_image_path) as source_image:
        img_pxsize = source_image.size
    img_pointsize = np.subtract(img_pxsize, 1)

    if ref_px == 'center, get from image':
        img_half = np.divide(img_pxsize, 2)
        img_third = np.divide(img_pxsize, 3)
        img_center_index = np.subtract(img_half, 0.5).tolist()
        ref_px = img_center_index

        x1 = -(img_half[0] - 0.5)
        x2 = -(img_third[0] / 2 - 0.5)
        x3 = img_third[0] / 2 - 0.5
        x4 = img_half[0] - 0.5
        y1 = img_half[1] - 0.5
        y2 = img_third[1] / 2 - 0.5
        y3 = -(img_third[1] / 2 - 0.5)
        y4 = -(img_half[1] - 0.5)
    
    img_grid_pxs = np.array([[x1,y1],[x2,y1],[x3,y1],[x4,y1],[x1,y2],[x2,y2],[x3,y2],[x4,y2],[x1,y3],[x2,y3],[x3,y3],[x4,y3],[x1,y4],[x2,y4],[x3,y4],[x4,y4]])
    img_TBLR_pxs = np.array([[0,y1],[0,y4],[x1,0],[x4,0]])

    return ref_px, img_grid_pxs, img_TBLR_pxs


def pxs_to_xyangs(AWIMtag_dictionary, img_dimensions, pxs):
    if isinstance(pxs, (list, tuple)): # models require numpy arrays
        pxs = np.asarray(pxs)

    input_shape = pxs.shape
    angs_direction = np.where(pxs < 0, -1, 1) # models are positive values only. Save sign. Same sign for xyangs

    pxs = np.abs(pxs).reshape(-1,2)

    if AWIMtag_dictionary['Pixel Angle Models Type'] == '3d_degree_poly_fit_abs_from_center':
        pxs_poly = np.zeros((pxs.shape[0], 9))
        pxs_poly[:,0] = pxs[:,0]
        pxs_poly[:,1] = pxs[:,1]
        pxs_poly[:,2] = np.square(pxs[:,0])
        pxs_poly[:,3] = np.multiply(pxs[:,0], pxs[:,1])
        pxs_poly[:,4] = np.square(pxs[:,1])
        pxs_poly[:,5] = np.power(pxs[:,0], 3)
        pxs_poly[:,6] = np.multiply(np.square(pxs[:,0]), pxs[:,1])
        pxs_poly[:,7] = np.multiply(pxs[:,0], np.square(pxs[:,1]))
        pxs_poly[:,8] = np.power(pxs[:,1], 3)

    xang_predict_coeff = AWIMtag_dictionary['Angles Model xang_coeffs']
    yang_predict_coeff = AWIMtag_dictionary['Angles Model yang_coeffs']
    coefficients_image_size = AWIMtag_dictionary['Angles Models Reference Dimensions']
    image_size_factor = max(img_dimensions) / max(coefficients_image_size)
    xang_predict_coeff = xang_predict_coeff / image_size_factor
    yang_predict_coeff = yang_predict_coeff / image_size_factor

    xyangs = np.zeros(pxs.shape)
    xyangs[:,0] = np.dot(pxs_poly, xang_predict_coeff)
    xyangs[:,1] = np.dot(pxs_poly, yang_predict_coeff)

    xyangs_pretty = np.multiply(xyangs.reshape(input_shape), angs_direction)

    return xyangs_pretty


# ----- unknown below this line -----
def xyangs_to_azarts(AWIMtag_dictionary, xyangs, ref_azart_override=False):

    # prepare to convert xyangs to azarts. Already have angs_direction from above. abs of xangs only, keep negative yangs
    input_shape = xyangs.shape
    angs_direction = np.where(xyangs < 0, -1, 1)
    xyangs = xyangs.reshape(-1,2)
    angs_direction = angs_direction.reshape(-1,2)
    xyangs[:,0] = np.abs(xyangs[:,0])
    xyangs *= math.pi/180
    if isinstance(ref_azart_override, (list, tuple, np.ndarray)):
        ref_azart_rad = np.multiply(ref_azart_override, math.pi/180)
    else:
        ref_azart_rad = np.multiply(AWIMtag_dictionary['Ref Pixel Azimuth Artifae'], math.pi/180)

    # see photoshop diagram of sphere, circles, and triangles for variable names
    xang_compliment = np.subtract(math.pi/2, xyangs[:,0]) # always (+) because xang < 90
    d1 = 1*np.cos(xang_compliment) # always (+)
    r2 = 1*np.sin(xang_compliment) # always (+)
    ang_totalsmallcircle = np.add(ref_azart_rad[1], xyangs[:,1]) # -180 to 180
    d2_ = np.multiply(np.cos(ang_totalsmallcircle), r2) # (-) for ang_totalsmallcircle > 90 or < -90, meaning px behind observer
    art_seg_ = np.multiply(np.sin(ang_totalsmallcircle), r2) # (-) for (-) ang_totalsmallcircle
    arts = np.arcsin(art_seg_) # (-) for (-) art_seg_
    az_rel = np.subtract(math.pi/2, np.arctan(np.divide(d2_, d1))) # d2 (-) for px behind observer and therefore az_rel > 90 because will subtract (-) atan
    az_rel = np.multiply(az_rel, angs_direction[:,0])
    azs = np.mod(np.add(ref_azart_rad[0], az_rel), 2*math.pi)

    azarts = np.zeros(xyangs.shape)
    azarts[:,0] = np.multiply(azs, 180/math.pi)
    azarts[:,1] = np.multiply(arts, 180/math.pi)

    azarts = azarts.reshape(input_shape)

    return azarts


# TODO this function is untested 1 jul 2022
def azarts_to_xyangs(AWIMtag_dictionary, azarts):
    if isinstance(azarts, list):
        azarts = np.asarray(azarts)

    input_shape = azarts.shape
    azarts = azarts.reshape(-1,2)

    ref_px_azart = AWIMtag_dictionary['Ref Pixel Azimuth Artifae']

    # find az_rels, convert to -180 < x <= 180, then abs value + direction matrix
    # also need az_rel compliment angle + store which are behind camera
    # then to radians
    simple_subtract = np.subtract(azarts[:,0], ref_px_azart[0])
    big_angle_correction = np.where(simple_subtract > 0, -360, 360)
    az_rel = np.where(np.abs(simple_subtract) <= 180, simple_subtract, np.add(simple_subtract, big_angle_correction))
    az_rel_direction = np.where(az_rel < 0, -1, 1)
    az_rel_abs = np.abs(az_rel)
    az_rel_behind_observer = np.where(az_rel_abs <= 90, False, True) # true if point is behind observer - assume because camera pointed up very high past zenith or pointed very low below nether-zenith
    # Note: cannot allow az_rel_compliment (and therefore d2) to be negative because must simple_ang_totalsmallcircle be (-) if art_seg_ is (-), which is good.
    az_rel_compliment = np.where(az_rel_abs <= 90, np.subtract(90, az_rel_abs), np.subtract(az_rel_abs, 90)) # 0 to 90 angle from line perpendicular to az
    az_rel_compliment_rad = np.multiply(az_rel_compliment, math.pi/180) # (+) only, 0 to 90
    ref_px_azart_rad = np.multiply(ref_px_azart, math.pi/180)

    # artifae direction matrix, then artifaes to radians, keep sign just convert to radian
    art_direction = np.where(azarts[:,1] < 0, -1, 1)
    art_rad = np.multiply(azarts[:,1], math.pi/180)

    # trigonometry, see photoshop diagrams for variable descriptions. segment ending with underscore_ means "can be negative distance"
    art_seg_ = np.sin(art_rad) # notice: (-) for (-) arts
    d3 = np.cos(art_rad) # (+) only, because arts are always -90 to 90
    d2 = np.multiply(np.sin(az_rel_compliment_rad), d3) # (+) only
    d1 = np.multiply(np.cos(az_rel_compliment_rad), d3) # (+) only because az_rel_compliment_rad is 0 to 90
    r2 = np.sqrt(np.square(d2), np.square(art_seg_))
    xang_abs = np.subtract(math.pi/2, np.arccos(d1)) # TODO? what if xang is actually > 90? would be unusual, difficult to combine with large yang
    xang = np.multiply(xang_abs, az_rel_direction)
    pt1_art_ = np.multiply(r2, np.sin(ref_px_azart_rad[1])) # (-) for (-) cam_arts, which is good
    lower_half = np.where(art_seg_ < pt1_art_, True, False) # true if px is below middle of photo
    ang_smallcircle_fromhorizon = np.arctan(np.divide(art_seg_, d2)) # -90 to 90, (-) for (-) art_seg_, bc d2 always (+)
    # for yang, if in front, simple, but behind observer, the angle from must be subtracted from 180 or -180 because different angle meaning see photoshop diagram
    ang_totalsmallcircle = np.where(np.logical_not(az_rel_behind_observer), ang_smallcircle_fromhorizon, np.subtract(np.multiply(art_direction, math.pi), ang_smallcircle_fromhorizon))
    yang = np.subtract(ang_totalsmallcircle, ref_px_azart_rad[1]) # simply subtract because |ang_totalsmallcircle| < 180 AND |center_azart[1]| < 90 AND if |ang_totalsmallcircle| > 90, then they are same sign

    xy_angs = np.zeros(input_shape)
    xy_angs[:,0] = np.multiply(xang, 180/math.pi)
    xy_angs[:,1] = np.multiply(yang, 180/math.pi)

    return xy_angs


# TODO this function is untested 1 jul 2022
def xyangs_to_pxs(AWIMtag_dictionary, xy_angs):

    input_shape = xy_angs.shape

    xy_angs_direction = np.where(xy_angs < 0, -1, 1)
    xy_angs_abs = np.abs(xy_angs)

    if AWIMtag_dictionary['Pixel Angle Models Type'] == '3d_degree_poly_fit_abs_from_center':
        xy_angs_poly = np.zeros((xy_angs.shape[0], 9))
        xy_angs_poly[:,0] = xy_angs_abs[:,0]
        xy_angs_poly[:,1] = xy_angs_abs[:,1]
        xy_angs_poly[:,2] = np.square(xy_angs_abs[:,0])
        xy_angs_poly[:,3] = np.multiply(xy_angs_abs[:,0], xy_angs_abs[:,1])
        xy_angs_poly[:,4] = np.square(xy_angs_abs[:,1])
        xy_angs_poly[:,5] = np.power(xy_angs_abs[:,0], 3)
        xy_angs_poly[:,6] = np.multiply(np.square(xy_angs_abs[:,0]), xy_angs_abs[:,1])
        xy_angs_poly[:,7] = np.multiply(xy_angs_abs[:,0], np.square(xy_angs_abs[:,1]))
        xy_angs_poly[:,8] = np.power(xy_angs_abs[:,1], 3)

    pxs = np.zeros(input_shape)
    px_models_df = AWIMtag_dictionary['Pixels Model']
    x_px_predict_coeff = px_models_df['']
    pxs[:,0] = np.dot(xy_angs_poly, self.x_px_predict_coeff)
    pxs[:,1] = np.dot(xy_angs_poly, self.y_px_predict_coeff)

    pxs = np.multiply(pxs, xy_angs_direction)

    return pxs


# TODO this function is untested 1 jul 2022
def px_coord_convert(input_pxs, input_type, output_type):
    if ('top-left' in input_type) and ('center' in output_type):
        pxs[:,0] = pxs[:,0] + self.center_KVpx[0]
        pxs[:,1] = pxs[:,1] + self.center_KVpx[1]


def ref_px_from_known_px(AWIMtag_dictionary, known_px, known_px_azart):
    xy_ang = pxs_to_xyangs(AWIMtag_dictionary, known_px)

    xy_ang *= -1

    ref_px_azart = xyangs_to_azarts(AWIMtag_dictionary, xy_ang, ref_azart_override=known_px_azart)

    return ref_px_azart


def pxs_to_azarts(AWIMtag_dictionary, pxs):
    xy_angs = pxs_to_xyangs(AWIMtag_dictionary, pxs)

    azarts = xyangs_to_azarts(AWIMtag_dictionary, xy_angs)

    return azarts


def get_pixel_sizes(AWIMtag_dictionary, dimensions):
    small_px = 10
    little_cross_LRUD = np.array([-small_px,0,small_px,0,0,small_px,0,-small_px]).reshape(-1,2)
    little_cross_angs = pxs_to_xyangs(AWIMtag_dictionary, dimensions, little_cross_LRUD)
    px_size_center_horizontal = (abs(little_cross_LRUD[0,0]) + abs(little_cross_LRUD[1,0])) / (abs(little_cross_angs[0,0]) + abs(little_cross_angs[1,0]))
    px_size_center_vertical = (abs(little_cross_LRUD[2,1]) + abs(little_cross_LRUD[3,1])) / (abs(little_cross_angs[2,1]) + abs(little_cross_angs[3,1]))

    border_angles = np.array(AWIMtag_dictionary['TBLR Angles'])
    horizontal_angle_width = abs(border_angles[2,0]) + abs(border_angles[3,0])
    vertical_angle_width = abs(border_angles[0,1]) + abs(border_angles[1,1])
    px_size_average_horizontal = (dimensions[0] - 1) / horizontal_angle_width
    px_size_average_vertical = (dimensions[1] - 1) / vertical_angle_width

    return [px_size_center_horizontal, px_size_center_vertical], [px_size_average_horizontal, px_size_average_vertical]


# put the AWIMtag in the comment field of the image exif and re-attach the exif to the image
def add_AWIMtag_to_exif():
	if img_exif_raw.get(37510):
		user_comments = img_exif_raw[37510]
	else:
		user_comments = ''
	img_exif_raw[37510] = user_comments + 'AWIMstart' + cam_AWIMtag_string + 'AWIMend'