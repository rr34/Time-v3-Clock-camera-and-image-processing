import math
import numpy as np
import PIL
import pandas as pd
import astropytools
import metadata_tools, formatters


def generate_empty_AWIMtag_dictionary(default_units=True):
    AWIMtag_dictionary = {}
    AWIMtag_dictionary['awim Location Coordinates'] = [-999.9, -999.9]
    AWIMtag_dictionary['awim Location Coordinates Unit'] = 'Latitude, Longitude; to 6 decimal places so ~11cm'
    AWIMtag_dictionary['awim Location Coordinates Source'] = ''
    AWIMtag_dictionary['awim Location MSL'] = -999.9
    AWIMtag_dictionary['awim Location MSL Unit'] = 'Photo meters above sea level; to 1 decimal place so 10cm'
    AWIMtag_dictionary['awim Location MSL Source'] = ''
    AWIMtag_dictionary['awim Location Terrain Elevation'] = -999.9
    AWIMtag_dictionary['awim Location Terrain Elevation Unit'] = 'Elevation of ground, meters above sea level; to 1 decimal place so 10cm'
    AWIMtag_dictionary['awim Location Terrain Elevation Source'] = ''
    AWIMtag_dictionary['awim Location AGL'] = -999.9
    AWIMtag_dictionary['awim Location AGL Unit'] = 'Meters above ground level; to 2 decimal places so 1cm'
    AWIMtag_dictionary['awim Location AGL Description'] = ''
    AWIMtag_dictionary['awim Location AGL Source'] = ''
    AWIMtag_dictionary['awim Capture Moment'] = '0000-01-01T00:00:00Z'
    AWIMtag_dictionary['awim Capture Moment Unit'] = 'Gregorian New Style Calendar in ISO 8601 YYYY-MM-DDTHH:MM:SSZ'
    AWIMtag_dictionary['awim Capture Moment Source'] = ''
    AWIMtag_dictionary['awim Models Type'] = '' # 3d_degree_poly_fit_abs_from_center
    AWIMtag_dictionary['awim Ref Pixel'] = [-999.9, -999.9]
    AWIMtag_dictionary['awim Ref Pixel Coord Type'] = 'top-left is (0,0) so standard; to 1 decimal so to tenth of a pixel'
    AWIMtag_dictionary['awim Ref Image Size'] = []
    AWIMtag_dictionary['awim Ref Image Size Note'] = 'awim tag contains ONLY the size of the original reference image of the camera calibration - not the particular instance of the image - for two reasons: 1. The digital image itself contains its own size, so metadata size would be duplicate information, 2. The user may use some other scaled size in practice. ONLY the original reference size is necessary and appropriate for metadata.'
    AWIMtag_dictionary['awim Angles Models Features'] = []
    AWIMtag_dictionary['awim Angles Model xang_coeffs'] = []
    AWIMtag_dictionary['awim Angles Model yang_coeffs'] = []
    AWIMtag_dictionary['awim Pixels Model Features'] = []
    AWIMtag_dictionary['awim Pixels Model xpx_coeffs'] = []
    AWIMtag_dictionary['awim Pixels Model ypx_coeffs'] = []
    AWIMtag_dictionary['awim Ref Pixel Azimuth Artifae'] = [-999.9, -999.9]
    AWIMtag_dictionary['awim Ref Pixel Azimuth Artifae Source'] = ''
    AWIMtag_dictionary['awim Ref Pixel Azimuth Artifae Unit'] = 'Degrees; to hundredth of a degree'
    AWIMtag_dictionary['awim Grid Pixels'] = []
    AWIMtag_dictionary['awim Grid Angles'] = []
    AWIMtag_dictionary['awim Grid Azimuth Artifae'] = []
    AWIMtag_dictionary['awim Grid RA Dec'] = []
    AWIMtag_dictionary['awim TBLR Pixels'] = []
    AWIMtag_dictionary['awim TBLR Angles'] = []
    AWIMtag_dictionary['awim TBLR Azimuth Artifae'] = []
    AWIMtag_dictionary['awim TBLR RA Dec'] = []
    AWIMtag_dictionary['awim RA Dec Unit'] = 'ICRS J2000 Epoch, to thousandth of an hour, hundredth of a degree'
    AWIMtag_dictionary['awim Pixel Size Center Horizontal Vertical'] = [-999.9, -999.9]
    AWIMtag_dictionary['awim Pixel Size Average Horizontal Vertical'] = [-999.9, -999.9]
    AWIMtag_dictionary['awim Pixel Size Unit'] = 'Pixels per Degree; to tenth of a pixel'

    if not default_units:
        AWIMtag_dictionary['awim Location Coordinates Unit'] = ''
        AWIMtag_dictionary['awim Location MSL Unit'] = ''
        AWIMtag_dictionary['awim Location Terrain Elevation Unit'] = ''
        AWIMtag_dictionary['awim Location Elevation Unit'] = ''
        AWIMtag_dictionary['awim Location AGL Unit'] = ''
        AWIMtag_dictionary['awim Capture Moment Unit'] = ''
        AWIMtag_dictionary['awim Ref Pixel Coord Type'] = ''
        AWIMtag_dictionary['awim Ref Pixel Azimuth Artifae Unit'] = ''
        AWIMtag_dictionary['awim RA Dec Unit'] = ''
        AWIMtag_dictionary['awim Pixel Size Unit'] = ''

    return AWIMtag_dictionary


def get_ref_px_thirds_grid_TBLR(source_image_path, ref_px):
    with PIL.Image.open(source_image_path) as source_image: # todo: do for jpg, not just png
        img_pxsize = source_image.size
    img_pointsize = np.subtract(img_pxsize, 1) # the size from pixel center to pixel center is 1 pixel smaller because it excludes all the edge pixels' outer halves

    img_half = np.divide(img_pxsize, 2)
    img_third = np.divide(img_pxsize, 3)
    img_center_index = np.subtract(img_half, 0.5).tolist()

    x1 = -(img_half[0] - 0.5)
    x2 = -(img_third[0] / 2 - 0.5)
    x3 = img_third[0] / 2 - 0.5
    x4 = img_half[0] - 0.5
    y1 = img_half[1] - 0.5
    y2 = img_third[1] / 2 - 0.5
    y3 = -(img_third[1] / 2 - 0.5)
    y4 = -(img_half[1] - 0.5)

    if ref_px == 'center, get from image': # todo: for ref_px allow for a cropped image where the reference pixel is not the center pixel
        ref_px = img_center_index

    img_grid_pxs = np.array([[x1,y1],[x2,y1],[x3,y1],[x4,y1],[x1,y2],[x2,y2],[x3,y2],[x4,y2],[x1,y3],[x2,y3],[x3,y3],[x4,y3],[x1,y4],[x2,y4],[x3,y4],[x4,y4]])
    img_TBLR_pxs = np.array([[0,y1],[0,y4],[x1,0],[x4,0]])

    return ref_px, img_grid_pxs, img_TBLR_pxs


def pxs_to_xyangs(AWIMtag_dictionary, pxs, imgsize_relative=1):
    if isinstance(pxs, (list, tuple)): # models require numpy arrays
        pxs = np.asarray(pxs)

    input_shape = pxs.shape
    angs_direction = np.where(pxs < 0, -1, 1) # models are positive values only. Save sign. Same sign for xyangs

    pxs = np.abs(pxs).reshape(-1,2)

    if AWIMtag_dictionary['awim Models Type'] == '3d_degree_poly_fit_abs_from_center':
        pxs = pxs / imgsize_relative
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

    xang_predict_coeff = AWIMtag_dictionary['awim Angles Model xang_coeffs']
    yang_predict_coeff = AWIMtag_dictionary['awim Angles Model yang_coeffs']

    xyangs = np.zeros(pxs.shape)
    xyangs[:,0] = np.dot(pxs_poly, xang_predict_coeff)
    xyangs[:,1] = np.dot(pxs_poly, yang_predict_coeff)

    xyangs_pretty = np.multiply(xyangs.reshape(input_shape), angs_direction)

    return xyangs_pretty


def closest_to_x_sides(guess_angle, oneside_angle, number_sides):
    sides_angles = []
    for x in range(0,number_sides):
        side_angle = oneside_angle + (x/number_sides)*360
        sides_angles.append(side_angle)

    differences = []
    for side_angle in sides_angles:
        difference = guess_angle - side_angle
        difference = (difference + 180) % 360 - 180
        differences.append(abs(difference))

    min_index = differences.index(min(differences))
    correct_angle = sides_angles[min_index]

    return correct_angle


def xyangs_to_azarts(AWIMtag_dictionary, xyangs, ref_azart_override=False):
    # prepare to convert xyangs to azarts. Already have angs_direction from above. abs of xangs only, keep negative yangs
    input_shape = xyangs.shape
    angs_direction = np.where(xyangs < 0, -1, 1)
    xyangs = xyangs.reshape(-1,2)
    angs_direction = angs_direction.reshape(-1,2)
    xyangs[:,0] = np.abs(xyangs[:,0])
    xyangs *= math.pi/180
    if isinstance(ref_azart_override, (list, tuple, np.ndarray)): # This gives the option to use the awim tag of any photo and point the photo in any direction.
        ref_azart_rad = np.multiply(ref_azart_override, math.pi/180)
    else:
        ref_azart_rad = np.multiply(AWIMtag_dictionary['awim Ref Pixel Azimuth Artifae'], math.pi/180)

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


def get_pixel_sizes(AWIMtag_dictionary, imgsize_relative=1):
    small_px = 10
    little_cross_LRUD = np.array([-small_px,0,small_px,0,0,small_px,0,-small_px]).reshape(-1,2)
    little_cross_angs = pxs_to_xyangs(AWIMtag_dictionary, little_cross_LRUD, imgsize_relative)
    px_size_center_horizontal = (abs(little_cross_LRUD[0,0]) + abs(little_cross_LRUD[1,0])) / (abs(little_cross_angs[0,0]) + abs(little_cross_angs[1,0]))
    px_size_center_vertical = (abs(little_cross_LRUD[2,1]) + abs(little_cross_LRUD[3,1])) / (abs(little_cross_angs[2,1]) + abs(little_cross_angs[3,1]))

    border_angles = np.array(AWIMtag_dictionary['awim TBLR Angles'])
    horizontal_angle_total = abs(border_angles[2,0]) + abs(border_angles[3,0])
    vertical_angle_total = abs(border_angles[0,1]) + abs(border_angles[1,1])
    border_pixels = np.array(AWIMtag_dictionary['awim TBLR Pixels'])
    horizontal_pixels = (abs(border_pixels[2,0]) + abs(border_pixels[3,0])) * imgsize_relative
    vertical_pixels = (abs(border_pixels[0,1]) + abs(border_pixels[1,1])) * imgsize_relative
    px_size_average_horizontal = horizontal_pixels / horizontal_angle_total
    px_size_average_vertical = vertical_pixels / vertical_angle_total

    return [px_size_center_horizontal, px_size_center_vertical], [px_size_average_horizontal, px_size_average_vertical]


# ----- unknown below this line -----
# TODO this function is untested 1 jul 2022
def azarts_to_xyangs(AWIMtag_dictionary, azarts):
    if isinstance(azarts, list):
        azarts = np.asarray(azarts)

    input_shape = azarts.shape
    azarts = azarts.reshape(-1,2)

    ref_px_azart = AWIMtag_dictionary['awim Ref Pixel Azimuth Artifae']

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


# imported from astronomical clock. Was part of an AstroImage object.
# get the px of an azalt in requested coord type (default is Kivy)
# calculation is 3 parts via azalt to 1. xy_angs to 2. px to 3. KVpx
def azalts_to_pxs(self, azalts, coord_type):
    if isinstance(azalts, list):
        azalts = np.asarray(azalts)

    input_shape = azalts.shape
    azalts = azalts.reshape(-1,2)

    # Part 1, azalts to xy_angs
    # find az_rels, convert to -180 < x <= 180, then abs value + direction matrix
    # also need az_rel compliment angle + store which are behind camera
    # then to radians
    simple_subtract = np.subtract(azalts[:,0], self.center_azalt[0])
    big_angle_correction = np.where(simple_subtract > 0, -360, 360)
    az_rel = np.where(np.abs(simple_subtract) <= 180, simple_subtract, np.add(simple_subtract, big_angle_correction))
    az_rel_direction = np.where(az_rel < 0, -1, 1)
    az_rel_abs = np.abs(az_rel)
    az_rel_behind_observer = np.where(az_rel_abs <= 90, False, True) # true if point is behind observer - assume because camera pointed up very high past zenith or pointed very low below nether-zenith
    # Note: cannot allow az_rel_compliment (and therefore d2) to be negative because must simple_ang_totalsmallcircle be (-) if alt_seg_ is (-), which is good.
    az_rel_compliment = np.where(az_rel_abs <= 90, np.subtract(90, az_rel_abs), np.subtract(az_rel_abs, 90)) # 0 to 90 angle from line perpendicular to az
    az_rel_compliment_rad = np.multiply(az_rel_compliment, math.pi/180) # (+) only, 0 to 90
    center_azalt_rad = np.multiply(self.center_azalt, math.pi/180)

    # altitude direction matrix, then altitudes to radians, keep sign just convert to radian
    alt_direction = np.where(azalts[:,1] < 0, -1, 1)
    alt_rad = np.multiply(azalts[:,1], math.pi/180)

    # trigonometry, see photoshop diagrams for variable descriptions. segment ending with underscore_ means "can be negative distance"
    alt_seg_ = np.sin(alt_rad) # notice: (-) for (-) alts
    d3 = np.cos(alt_rad) # (+) only, because alts are always -90 to 90
    d2 = np.multiply(np.sin(az_rel_compliment_rad), d3) # (+) only
    d1 = np.multiply(np.cos(az_rel_compliment_rad), d3) # (+) only because az_rel_compliment_rad is 0 to 90
    r2 = np.sqrt(np.square(d2), np.square(alt_seg_))
    xang_abs = np.subtract(math.pi/2, np.arccos(d1)) # TODO? what if xang is actually > 90? would be unusual, difficult to combine with large yang
    xang = np.multiply(xang_abs, az_rel_direction)
    pt1_alt_ = np.multiply(r2, np.sin(center_azalt_rad[1])) # (-) for (-) cam_alts, which is good
    lower_half = np.where(alt_seg_ < pt1_alt_, True, False) # true if px is below middle of photo
    ang_smallcircle_fromhorizon = np.arctan(np.divide(alt_seg_, d2)) # -90 to 90, (-) for (-) alt_seg_, bc d2 always (+)
    # for yang, if in front, simple, but behind observer, the angle from must be subtracted from 180 or -180 because different angle meaning see photoshop diagram
    ang_totalsmallcircle = np.where(np.logical_not(az_rel_behind_observer), ang_smallcircle_fromhorizon, np.subtract(np.multiply(alt_direction, math.pi), ang_smallcircle_fromhorizon))
    yang = np.subtract(ang_totalsmallcircle, center_azalt_rad[1]) # simply subtract because |ang_totalsmallcircle| < 180 AND |center_azalt[1]| < 90 AND if |ang_totalsmallcircle| > 90, then they are same sign

    xy_angs = np.zeros(azalts.shape)
    xy_angs[:,0] = np.multiply(xang, 180/math.pi)
    xy_angs[:,1] = np.multiply(yang, 180/math.pi)

    # Part 2: xy_angs to pxs
    xy_angs_direction = np.where(xy_angs < 0, -1, 1)
    xy_angs_abs = np.abs(xy_angs)

    if self.pixel_map_type == '3d_degree_poly_fit_abs_from_center':
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

    pxs = np.zeros(azalts.shape)
    pxs[:,0] = np.dot(xy_angs_poly, self.x_px_predict_coeff)
    pxs[:,1] = np.dot(xy_angs_poly, self.y_px_predict_coeff)

    pxs = np.multiply(pxs, xy_angs_direction)

    # Part 3. pxs to coord_type
    if coord_type == 'KVpx':
        pxs[:,0] = pxs[:,0] + self.center_KVpx[0]
        pxs[:,1] = pxs[:,1] + self.center_KVpx[1]

    return pxs


    def pxs_to_azalts(self, pxs):
        if isinstance(pxs, list): # models require numpy arrays
            pxs = np.asarray(pxs)

        input_shape = pxs.shape
        angs_direction = np.where(pxs < 0, -1, 1) # models are positive values only. Save sign. Same sign for xyangs
        pxs = np.abs(pxs)
        pxs = pxs.reshape(-1,2)

        if self.pixel_map_type == '3d_degree_poly_fit_abs_from_center':
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

        xyangs = np.zeros(pxs.shape)
        xyangs[:,0] = np.dot(pxs_poly, self.xang_predict_coeff)
        xyangs[:,1] = np.dot(pxs_poly, self.yang_predict_coeff)

        xyangs_pretty = np.multiply(xyangs.reshape(input_shape), angs_direction)

        # prepare to convert xyangs to azalts. Already have angs_direction from above. abs of xangs only, keep negative yangs
        angs_direction = angs_direction.reshape(-1,2)
        xyangs = xyangs_pretty.reshape(-1,2)
        xyangs[:,0] = np.abs(xyangs[:,0])
        xyangs = np.multiply(xyangs, math.pi/180)
        center_azalt_rad = np.multiply(self.center_azalt, math.pi/180)

        # see photoshop diagram of sphere, circles, and triangles for variable names
        xang_compliment = np.subtract(math.pi/2, xyangs[:,0]) # always (+) because xang < 90
        d1 = 1*np.cos(xang_compliment) # always (+)
        r2 = 1*np.sin(xang_compliment) # always (+)
        ang_totalsmallcircle = np.add(center_azalt_rad[1], xyangs[:,1]) # -180 to 180
        d2_ = np.multiply(np.cos(ang_totalsmallcircle), r2) # (-) for ang_totalsmallcircle > 90 or < -90, meaning px behind observer
        alt_seg_ = np.multiply(np.sin(ang_totalsmallcircle), r2) # (-) for (-) ang_totalsmallcircle
        alts = np.arcsin(alt_seg_) # (-) for (-) alt_seg_
        az_rel = np.subtract(math.pi/2, np.arctan(np.divide(d2_, d1))) # d2 (-) for px behind observer and therefore az_rel > 90 because will subtract (-) atan
        az_rel = np.multiply(az_rel, angs_direction[:,0])
        azs = np.mod(np.add(center_azalt_rad[0], az_rel), 2*math.pi)

        azalts = np.zeros(xyangs.shape)
        azalts[:,0] = np.multiply(azs, 180/math.pi)
        azalts[:,1] = np.multiply(alts, 180/math.pi)

        return azalts.reshape(input_shape)

