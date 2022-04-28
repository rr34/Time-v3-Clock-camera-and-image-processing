from io import StringIO
import math
import numpy as np
import pandas as pd
import datetime


class ImageAWIMData(object):

    def __init__(self, awim_dictionary):

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