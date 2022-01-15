from matplotlib import image
import numpy as np
import pandas as pd
from numpy.core.function_base import linspace
from pandas.core.frame import DataFrame
from sklearn import linear_model

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
	def __init__(self, camera_name=None, image_width=None, image_height=None, lens_name=None,
				 zoom=None, settings_notes=None, aperture=None, reference_pixels=None,
				 source_file_location=None, ref_data_type="whole_image", pixel_aim=None):
				 
		"""
        Parameters
        ----------
		camera_name : the name of the camera. Can include some lens and settings notes for quick
			identification
			
		image_width : the width of the camera's sensor in pixels. This should be the maximum
			resolution of the camera.
		
		image_height : the height of the camera's sensor in pixels. This should be the maximum
			resolution of the camera. Landscape is the stadard so the height will normally be
			less than the width.
		
		lens_name, zoom, settings_notes : (optional) specifying these is optional, but each is
			critical to the image properties from a particular camera.
			
		aperture : (optional) the aperture should not affect the angles of the pixels, but is
			here for completeness as desired.
			
		reference_pixels : either reference_pixels or source_file_location with reference pixels
			must be specified in order to define the properties of the camera.
			
		source_file_location : (optional) camera properties should be able to be stored for quick
			future use.
			
		"""
 
	
		self.camera_name = camera_name
		self.image_width = image_width
		self.image_height = image_height
		self.max_x_ref = image_width-1
		self.max_y_ref = image_height-1
		self.center_x_ref = (self.max_x_ref / 2)
		self.center_y_ref = (self.max_y_ref / 2)
		self.lens_name = lens_name
		self.zoom = zoom
		self.settings_notes = settings_notes
		self.aperture = aperture
		self.source_file_location = source_file_location
		self.reference_pixels_df = pd.read_csv(source_file_location)
		self.ref_data_type=ref_data_type
		self.dataframe_columns = ['px_x', 'px_y', 'alt', 'az']


	# TODO def __repr__(self):
		"""
		readable representation of the '~camera.CameraAim' object
		"""

	
	# @classmethod ??
	def angles_array_extrapolation(self):

		reference_pixels_df = self.reference_pixels_df

		# make the center pixel the zero pixel
		"""
		reference_pixels_df['px_x'] = reference_pixels_df['px_x'] - self.center_x_ref
		reference_pixels_df['px_y'] = reference_pixels_df['px_y'] - self.center_y_ref
		"""

		# create the model to predict the rest of the pixels' alt
		independent_vars = reference_pixels_df[['px_x', 'px_y']] # becomes a dataframe with just the two labeled columns
		dependent_vars = reference_pixels_df['alt']
		alt_model = linear_model.LinearRegression()
		alt_model.fit(independent_vars, dependent_vars)

		# create the "meshgrid" dataframe for the entire image
		camera_aim_df = DataFrame(columns=self.dataframe_columns)
		print(camera_aim_df)
		for x in np.arange(step=100, stop=self.max_x_ref):
			for y in np.arange(step=100, stop=self.max_y_ref):
				camera_aim_df.loc[len(camera_aim_df.index)] = [x, y, np.nan, np.nan]

		# fill in the meshgrid dataframe with predicted values
		camera_aim_df['alt'] = alt_model.predict(camera_aim_df[['px_x', 'px_y']])

		return camera_aim_df
		
		"""
		# first, if the data is for one quadrant, complete the reference pixel dataframe by mirroring
		if self.ref_data_type == "UL_quadrant":
			UR_quadrant_df = DataFrame(reference_pixels_df.values, columns=reference_pixels_df.columns)
			UR_quadrant_df['px_x'] = UR_quadrant_df['px_x'] * -1 + 2 * self.center_x_ref
			UR_quadrant_df['px_y'] = UR_quadrant_df['px_y']
			UR_quadrant_df['alt'] = UR_quadrant_df['alt']
			UR_quadrant_df['az'] = UR_quadrant_df['az'] * -1
			reference_pixels_df = reference_pixels_df.append(UR_quadrant_df, ignore_index=True)
			bottom_half_df = DataFrame(reference_pixels_df.values, columns=reference_pixels_df.columns)
			bottom_half_df['px_x'] = bottom_half_df['px_x']
			bottom_half_df['px_y'] = bottom_half_df['px_y'] * -1 + 2 * self.center_y_ref
			bottom_half_df['alt'] = bottom_half_df['alt'] * -1
			bottom_half_df['az'] = bottom_half_df['az']
			reference_pixels_df = reference_pixels_df.append(bottom_half_df, ignore_index=True)




		# start working with numpy arrays here, create the 3D array with meshgrids for first two layers
		pixel_aim = np.empty(shape=(self.image_height, self.image_width, 4)) # see note above. Shape is row COUNT x column COUNT! NOT "row width x column width". Layers are x values, y values alt values, az values
		x = np.linspace(0, self.image_width-1, self.image_width)
		y = np.linspace(0, self.image_height-1, self.image_height)
		pixel_aim[:,:,0], pixel_aim[:,:,1] = np.meshgrid(x, y)

		return pixel_aim
	"""