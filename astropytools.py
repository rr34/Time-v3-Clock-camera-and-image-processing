import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, get_sun, get_moon, get_body, solar_system_ephemeris
import awimlib

def get_AzArt(AWIMtag_dictionary, celestial_object):
    capture_moment = awimlib.format_datetime(AWIMtag_dictionary['CaptureMoment'], 'from string')
    earth_latlng = AWIMtag_dictionary['Location']

    astropy_moment = Time(capture_moment)
    img_astropy_location = EarthLocation(lat=earth_latlng[0]*u.deg, lon=earth_latlng[1]*u.deg)
    img_astropy_AltAzframe = AltAz(obstime=astropy_moment, location=img_astropy_location)
    if celestial_object == 'sun':
        object_SkyCoords = get_sun(astropy_moment)
    elif celestial_object == 'moon':
        object_SkyCoords = get_moon(astropy_moment)
    elif celestial_object in ['mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']:
        object_SkyCoords = get_body(celestial_object, astropy_moment)
    
    object_AltAz = object_SkyCoords.transform_to(img_astropy_AltAzframe)

    known_azart = [object_AltAz.az.degree, object_AltAz.alt.degree]

    return known_azart


def AzArts_to_RADecs(AWIMtag_dictionary, azarts):
    input_shape = azarts.shape
    azarts = azarts.reshape(-1,2)
    capture_moment = awimlib.format_datetime(AWIMtag_dictionary['CaptureMoment'], 'from string')
    earth_latlng = AWIMtag_dictionary['Location']

    astropy_moment = Time(capture_moment)
    img_astropy_location = EarthLocation(lat=earth_latlng[0]*u.deg, lon=earth_latlng[1]*u.deg)
    astropy_AltAzsframe = AltAz(obstime=astropy_moment, location=img_astropy_location, az=azarts[:,0]*u.deg, alt=azarts[:,1]*u.deg)
    astropy_SkyCoord = SkyCoord(astropy_AltAzsframe)

    RADecs = np.ndarray(azarts.shape)

    RADecs[:,0] = astropy_SkyCoord.icrs.ra.hourangle
    RADecs[:,1] = astropy_SkyCoord.icrs.dec.degree

    RADecs = RADecs.reshape(input_shape)

    return RADecs