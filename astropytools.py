import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon, get_body, solar_system_ephemeris
import awimlib

def get_AzArt(AWIMtag_dictionary, celestial_object):
    capture_moment = awimlib.format_datetime(AWIMtag_dictionary['CaptureMoment'], 'from string')
    earth_latlng = AWIMtag_dictionary['Location']

    # Astropy starts here
    img_astropy_time = Time(capture_moment)
    img_astropy_location = EarthLocation(lat=earth_latlng[0]*u.deg, lon=earth_latlng[1]*u.deg)
    img_astropy_AltAzframe = AltAz(obstime=img_astropy_time, location=img_astropy_location)
    if celestial_object == 'sun':
        object_SkyCoords = get_sun(img_astropy_time)
    elif celestial_object == 'moon':
        object_SkyCoords = get_moon(img_astropy_time)
    elif celestial_object in ['mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']:
        object_SkyCoords = get_body(celestial_object, img_astropy_time)
    
    object_AltAz = object_SkyCoords.transform_to(img_astropy_AltAzframe)

    known_azart = [object_AltAz.az.degree, object_AltAz.alt.degree]

    return known_azart


def AzArt_to_RADec():
    pass