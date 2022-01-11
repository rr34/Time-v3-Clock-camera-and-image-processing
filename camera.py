import numpy as np

class CameraAim(object)
	"""
	A container classfor information about the properties of a particular
	camera / lens / settings system, especially the direction of each pixel
	or manageable fraction of the pixels relative to the a reference pixel.
	CameraAim is specific to a camera with a specific lens attached and
	set to specific settings
	"""
	def __init__(self, camera_name=None, image_width=None, image_height=None, lens_name=None,
				 zoom=None, settings_notes=None, aperture=None, reference_pixels=None,
				 source_file_location=None):
				 
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
		self.lens_name = lens_name
		self.zoom = zoom
		self.settings_notes = settings_notes
		self.aperture = aperture
		self.reference_pixels = reference_pixels
		self.source_file_location = source_file_location
	
	#TODO def __repr__(self):
	
	def angles_array_linear_extrapolation(self):
		pixel_aim_alt = np.zeros(shape=(self.image_width,self.image_height))
		pixel_aim_az = np.zeros(shape=(self.image_width,self.image_height))
		
		for this_row in self.reference_pixels
			pixel_aim_alt[this_row[0],this_row[1]] = this_row[3]
			pixel_aim_az[this_row[0],this_row[1]] = this_row[4]