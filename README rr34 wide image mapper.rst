# AstroWideImageMapper
 The AstroWideImageMapper project aims to enable users to label "wide-angle" images with pixel-by-pixel astronomical coordinate data.
 "Wide-angle" specifically means NOT a telescope where the image can be assumed to be flat but rather any camera like a point-and-shoot, smartphone, etc.
 where the direction of all the pixels cannot be extrapolated by one or two refrence pixels and a constant "pixel delta."
 In order to map the pixels of a wide-angle image, the properties of the camera and lens system must be known in more detail along with sufficiently specific settings data.
 There must be a standard way to define and manipulate such data. It will be useful to enable the user to gather tens or hundreds of pixel direction data from a
 camera / lens / settings system one time and after gathering a manageable number of data points let the software fill in an estimation for the rest of the millions of pixels
 gathered by modern commercially-available cameras - or some useful fraction thereof.
 The extreme camera example would be, for example, a fish-eye lens where image distortion is purposeful and extreme, obviously not flat, the opposite of a telescope.
 
 Some assumptions (that I believe are valid) make the challenge possible:
 1. Any camera, where the lens and settings are specifically defined, points its pixel sensors in specific non-changing direction relative to the other pixels.
 2. If the location (on Earth or the surface of any cellestial object) and orientation of the camera at the time of the image collection is known,
	each pixel can be assigned an altitude/azimuth coordinate relative to the cellestial object that does not change over time.
 
 The following are required:
 
 A new container class, CameraAim, to define the properties of a particular camera / lens / settings system, especially the direction of each pixel or manageable fraction
 of the pixels relative to the a reference pixel.
 CameraAim is specific to a camera with a specific lens and specific settings.
 
 A new function(?) / method(?) within CameraAim to combine the information from
 1. the CameraAim class
 2. astroplan.Observer class
 3. orientation data
 4. the capture moment of an image
 and return a class "Snapshot" that for a particular image that includes
 1. astropy.coordinates.builtin_frames.AltAz object with NumPy arrays for altitude and azimuth that correspond to the altitude and azimuth of
 each pixel in the image - or some manageable and useful fraction of the pixels.
 2. capture moment of the image
 3. name of camera.
 4. etc.
 
 A new function within CameraAim that efficiently saves CameraAim data to disk for future use.
 
 A new function within CameraAim that 
 
 Humans are good at orienting themselves with respect to 2-dimensional spaces we can see and travel within. The cosmos being largely not visible and 3-dimensional
 and being that we only travel in a repeating relatively tiny ellipse throughout our lives and being that we constantly rotate and we constantly orbit around
 the brightest light within said space, astonomers are accustomed to maintaining a mental model of the cosmos and lay persons are accustomed to being essentially lost within it.
 Software is often used to cover up challenging questions with instant answers and enable humans to un-know things.
 Software can also be used to shine light on questions and enable humans to challenge themselves to ask ask questions previously thought un-knowable by shining light on
 questions rather than the answers.

 Coding data standardization notes:
 - Any coordinate values are stored as a list of two floats with the horizontal value first, i.e. [azimuth, altitude], [x pixel, y pixel], [latitude, longitude]
 - for lists of coordinates, store as a 2-dim, two-column Numpy array OR a 3-dim Numpy array such that the 0-layer is horizontal, 1-layer is vertical
 - All time information is stored as  "aware" datetime object where date is Gregorian NS, time is UTC. datetime tzinfo source is datetime.timezone.UTC
