from io import StringIO
import math
import numpy as np
import pandas as pd
from PIL import Image, PngImagePlugin
import datetime


class ImageAstroData(object):

    """
	A container class for information about the properties of a particular
	image, especially the direction of each pixel
	ImageAWIM is specific to an individual image and thus has an ID retrieved from the image file to associate it with an image

	Note, TODO pixels are referenced to the center. The center is given in Photoshop reference then all pixel references
    after are relative to the center with right and up as positive.
    TODO
	i.e. (x, y), top-left to bottom-right:
	e.g. the top-left pixel is (0, 0), top-right is (1919, 0), bottom-left is (0, 1079), bottom-right is (1919, 1079)
	"""
    # TODO
    def __init__(self, awim_dictionary):
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
        
        self.latlng = [float(value) for value in (awim_dictionary['Location']).split(',')]
        self.capture_moment = datetime.datetime.fromisoformat(awim_dictionary['Capture Moment'])
        self.dimensions = [int(value) for value in (awim_dictionary['Dimensions']).split(',')]
        self.center_px = [float(value) for value in (awim_dictionary['Center Pixel']).split(',')]
        self.center_azalt = [float(value) for value in (awim_dictionary['Center AzAlt']).split(',')]
        px_models_df = pd.read_csv(StringIO(awim_dictionary['Pixel Models']), index_col=0)
        self.px_predict_features = px_models_df.columns
        self.x_px_predict_coeff = px_models_df.loc[['x_px_predict']].values[0]
        self.y_px_predict_coeff = px_models_df.loc[['y_px_predict']].values[0]
        self.pixel_map_type = awim_dictionary['Pixel Map Type']
        xyangs_models_df = pd.read_csv(StringIO(awim_dictionary['x,y Angle Models']), index_col=0)
        self.ang_predict_features = xyangs_models_df.columns
        self.xang_predict_coeff = xyangs_models_df.loc[['xang_predict']].values[0]
        self.yang_predict_coeff = xyangs_models_df.loc[['yang_predict']].values[0]
        # self.px_borders = [[float(value.split(' ')[0]), float(value.split(' ')[1])] for value in (awim_dictionary['Pixel Borders']).split(',')]
        # self.xyangs_borders = [float(value) for value in (awim_dictionary['x,y Angle Borders']).split(',')]