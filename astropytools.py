import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon, get_body, solar_system_ephemeris, ICRS
import formatters


def get_AzArt(AWIMtag_dictionary, celestial_object):
    capture_moment = formatters.format_datetime_old(AWIMtag_dictionary['Capture Moment'], 'from AWIM string')
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


# imported from AWIM2
def get_AzArts(earth_latlng, moments, celestial_object):
    astropy_moments = Time(moments)
    img_astropy_location = EarthLocation(lat=earth_latlng[0]*u.deg, lon=earth_latlng[1]*u.deg)
    img_astropy_AltAzframe = AltAz(obstime=astropy_moments, location=img_astropy_location)
    if celestial_object == 'sun':
        object_SkyCoords = get_sun(astropy_moments)
    elif celestial_object in ['moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']:
        object_SkyCoords = get_body(celestial_object, astropy_moments)
    
    object_AltAz = object_SkyCoords.transform_to(img_astropy_AltAzframe)

    azimuths = [x.az.degree for x in object_AltAz]
    artifaes = [x.alt.degree for x in object_AltAz]

    # known_azart = [object_AltAz.az.degree, object_AltAz.alt.degree]

    return azimuths, artifaes


def AzArts_to_RADecs(AWIMtag_dictionary, azarts):
    input_shape = azarts.shape
    azarts = azarts.reshape(-1,2)
    capture_moment = formatters.format_datetime_old(AWIMtag_dictionary['Capture Moment'], 'from AWIM string')
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


# imported from AWIM2
def day_night_twilight(sun_arts, moon_arts):
    tags_list = []
    light = ''

    sun_movement = sun_arts[1] - sun_arts[0]
    moon_movement = moon_arts[1] - moon_arts[0]
    day_count = 0
    night_count = 0
    ECT_count = 0
    ENT_count = 0
    EAT_count = 0
    MAT_count = 0
    MNT_count = 0
    MCT_count = 0
    for i in range(0, len(sun_arts)):
        tag_string = ''
        if 2 <= i <= len(sun_arts):
            sun_movement_last = sun_movement
            moon_movement_last = moon_movement
            sun_movement = sun_arts[i] - sun_arts[i-1]
            moon_movement = moon_arts[i] - moon_arts[i-1]
            if sun_arts[i] > 0 and sun_movement_last > 0 and sun_movement <= 0:
                tag_string += 'sunnoon,'
            if sun_arts[i] < 0 and sun_movement_last < 0 and sun_movement >= 0:
                tag_string += 'midnight,'
            if 'MCT' in tags_list[-2] and 'day' in tags_list[-1]:
                tag_string += 'sunrise,'
            if 'day' in tags_list[-2] and 'ECT' in tags_list[-1]:
                tag_string += 'sunset,'
            if moon_arts[i-1] <= 0 and moon_arts[i] > 0:
                tag_string += 'moonrise,'
            if moon_arts[i-1] > 0 and moon_arts[i] <= 0:
                tag_string += 'moonset,'
            if moon_arts[i] > 0 and moon_movement_last > 0 and moon_movement <= 0:
                tag_string += 'moonnoon,'
            
        if sun_arts[i] >= -0.833 and sun_movement >= 0:
            if 'day' not in light:
                day_count +=1
            light = 'day' + str(day_count)
        elif sun_arts[i] < -18:
            if 'night' not in light:
                night_count +=1
            light = 'night' + str(night_count)
        elif sun_movement < 0 and -0.833 > sun_arts[i] >= -6:
            if 'ECT' not in light:
                ECT_count +=1
            light = 'ECT' + str(ECT_count)
        elif sun_movement < 0 and -6 > sun_arts[i] >= -12:
            if 'ENT' not in light:
                ENT_count +=1
            light = 'ENT' + str(ENT_count)
        elif sun_movement < 0 and -12 > sun_arts[i] >= -18:
            if 'EAT' not in light:
                EAT_count +=1
            light = 'EAT' + str(EAT_count)
        elif sun_movement > 0 and -0.833 > sun_arts[i] >= -6:
            if 'MCT' not in light:
                MCT_count +=1
            light = 'MCT' + str(MCT_count)
        elif sun_movement > 0 and -6 > sun_arts[i] >= -12:
            if 'MNT' not in light:
                MNT_count +=1
            light = 'MNT' + str(MNT_count)
        elif sun_movement > 0 and -12 > sun_arts[i] >= -18:
            if 'MAT' not in light:
                MAT_count +=1
            light = 'MAT' + str(MAT_count)
        
        tag_string += light
        tags_list.append(tag_string)

    return tags_list