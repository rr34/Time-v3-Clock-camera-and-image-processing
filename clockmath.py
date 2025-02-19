from io import StringIO
import PIL
from PIL.ImageFilter import BoxBlur
from datetime import timezone
import math
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon, get_body
import astroimage
import astroplan

# this function takes a long time to run, greatly affected by the value of gridpts, but is valid for something like 24 hours
def calculate_astro_risesandsets(moment_now, earth_latlng, elevation):
    time_now_astropy = Time(moment_now)
    clock_astroplan_observer = astroplan.Observer(longitude=earth_latlng[1]*u.deg, latitude=earth_latlng[0]*u.deg, elevation=elevation*u.m, name='Time v3 Clock', timezone=timezone.utc)
    gridpts = 150

    print('calculating sun daily events')
    sun_daily = np.empty(12, dtype=np.dtype('datetime64[ns]'))
    thirteen_hours = np.timedelta64(13, 'h')

    sun_daily[0] = clock_astroplan_observer.sun_rise_time(time=time_now_astropy, which='previous', horizon=-0.833*u.deg, n_grid_points=gridpts).datetime64
    sun_daily[1] = clock_astroplan_observer.sun_rise_time(time=time_now_astropy, which='next', horizon=-0.833*u.deg, n_grid_points=gridpts).datetime64
    sun_daily[2] = clock_astroplan_observer.sun_rise_time(time=Time(sun_daily[1]+thirteen_hours), which='next', horizon=-0.833*u.deg, n_grid_points=gridpts).datetime64
    sun_daily[3] = clock_astroplan_observer.noon(time=time_now_astropy, which='previous', n_grid_points=gridpts).datetime64
    sun_daily[4] = clock_astroplan_observer.noon(time=time_now_astropy, which='next', n_grid_points=gridpts).datetime64
    sun_daily[5] = clock_astroplan_observer.noon(time=Time(sun_daily[4]+thirteen_hours), which='next', n_grid_points=gridpts).datetime64
    sun_daily[6] = clock_astroplan_observer.sun_set_time(time=time_now_astropy, which='previous', horizon=-0.833*u.deg, n_grid_points=gridpts).datetime64
    sun_daily[7] = clock_astroplan_observer.sun_set_time(time=time_now_astropy, which='next', horizon=-0.833*u.deg, n_grid_points=gridpts).datetime64
    sun_daily[8] = clock_astroplan_observer.sun_set_time(time=Time(sun_daily[7]+thirteen_hours), which='next', horizon=-0.833*u.deg, n_grid_points=gridpts).datetime64
    sun_daily[9] = clock_astroplan_observer.midnight(time=time_now_astropy, which='previous', n_grid_points=gridpts).datetime64
    sun_daily[10] = clock_astroplan_observer.midnight(time=time_now_astropy, which='next', n_grid_points=gridpts).datetime64
    sun_daily[11] = clock_astroplan_observer.midnight(time=Time(sun_daily[10]+thirteen_hours), which='next', n_grid_points=gridpts).datetime64

    print('calculating moon daily events')
    moon_daily = np.empty(6, dtype=np.dtype('datetime64[ns]'))
    moon_daily[0] = clock_astroplan_observer.moon_rise_time(time=time_now_astropy, which='previous', horizon=0*u.deg, n_grid_points=gridpts).datetime64
    moon_daily[1] = clock_astroplan_observer.moon_rise_time(time=time_now_astropy, which='next', horizon=0*u.deg, n_grid_points=gridpts).datetime64
    moon_daily[2] = clock_astroplan_observer.moon_rise_time(time=Time(moon_daily[1]+thirteen_hours), which='next', horizon=0*u.deg, n_grid_points=gridpts).datetime64
    moon_daily[3] = clock_astroplan_observer.moon_set_time(time=time_now_astropy, which='previous', horizon=0*u.deg, n_grid_points=gridpts).datetime64
    moon_daily[4] = clock_astroplan_observer.moon_set_time(time=time_now_astropy, which='next', horizon=0*u.deg, n_grid_points=gridpts).datetime64
    moon_daily[5] = clock_astroplan_observer.moon_set_time(time=Time(moon_daily[4]+thirteen_hours), which='next', horizon=0*u.deg, n_grid_points=gridpts).datetime64

    return sun_daily, moon_daily


# this can be calculated for each moment to the minute and cross over at sun midnight if the big astropy calcs are completed first.
def calculate_astro_daynightlength(moment_now, sun_daily):
    # todo: decide what day it is using sun midnight.
    # todo: update these variables to come from the function parameters
    print('calculating day / night length')
    nearest_sunrise_astropy = clock_astroplan_observer.sun_rise_time(time=time_now_astropy, which='nearest', horizon=-0.833*u.deg, n_grid_points=gridpts)
    next_sunset = clock_astroplan_observer.sun_set_time(time=nearest_sunrise_astropy, which='next', horizon=-0.833*u.deg, n_grid_points=gridpts).datetime64
    day_length = next_sunset - nearest_sunrise_astropy.datetime64
    night_length = np.timedelta64(24, 'h') - day_length
    day_night_length_str = '%i:%.2i day length / %i:%.2i night length.' % (np.timedelta64(day_length, 'h').astype(int), np.timedelta64(day_length, 'm').astype(int)%60, np.timedelta64(night_length, 'h').astype(int), np.timedelta64(night_length, 'm').astype(int)%60)

    return day_night_length_str

# dictionary of objects, get data for objects at moments, return dictionary
# for each celestial object in dictionary, numpy array
# each array row: [0 moment, 1 az, 2 alt, 3 ra, 4 dec, 5 distance from earth number
def calculate_astro_data(moments, celestial_objects_list, earth_latlng):
    celestial_objs_dictionary = {}
    img_astropy_location = EarthLocation(lat=earth_latlng[0]*u.deg, lon=earth_latlng[1]*u.deg) # can be outside loop because photos are near each other and using same latlng for all
    img_astropy_times = Time(moments)
    img_astropy_altazframes = AltAz(obstime=img_astropy_times, location=img_astropy_location)
    planets_list = ['mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']
    for celestial_object in celestial_objects_list:
        if celestial_object == 'sun':
            object_SkyCoords = get_sun(img_astropy_times)
            object_type = 'sun'
        elif celestial_object == 'moon':
            object_SkyCoords = get_moon(img_astropy_times)
            object_type = 'moon'
        elif any(str in celestial_object for str in planets_list):
            object_SkyCoords = get_body(celestial_object, img_astropy_times)
            object_type = 'planet'
        else:
            object_SkyCoords = SkyCoord.from_name('alnilam')
            object_type = 'star'

        object_AltAzs = object_SkyCoords.transform_to(img_astropy_altazframes)

        celestial_object_azs = object_AltAzs.az.degree
        celestial_object_alts = object_AltAzs.alt.degree
        celestial_object_ras = object_SkyCoords.ra.degree
        celestial_object_decs = object_SkyCoords.dec.degree
        if object_type != 'star':
            celestial_object_distances = object_SkyCoords.distance.au
        else:
            celestial_object_distances = np.full(moments.size, 0)

        astro_data = np.zeros((moments.size, 6))
        # astro_data[:,0] = moments # note: dtype not the same. this converts datetime64 objects into float
        astro_data[:,1] = celestial_object_azs
        astro_data[:,2] = celestial_object_alts
        astro_data[:,3] = celestial_object_ras
        astro_data[:,4] = celestial_object_decs
        astro_data[:,5] = celestial_object_distances

        celestial_objs_dictionary[celestial_object] = astro_data
    
    return celestial_objs_dictionary


# output dictionary of available images
# each image with a dictionary of celestial objects
# each object with a numpy array of placement data by moment# for a single location, list of moments, images, celestial objects
# 1. return a numpy array of standard astro data by moment
# 2. and a dictionary of the images, each with a dictionary of celestial objects, each with a numpy array of image-specific data
# the data in the rows of the numpy arrays correspond to the moments in column 0 of the astro data array
def calculate_astro_data_to_images(moments, celestial_objs_dictionary, awim_imgs_list, img_dims, sun_moon_size_px):
    imgs_objs_dictionary = {}
    tracker_list = []
    for awim_filename in awim_imgs_list:
        imgs_objs_dictionary[awim_filename] = {} # bc dictionary of dictionaries
        awim_png = PIL.Image.open(awim_filename)
        awim_dictionary_text = awim_png.text
        awim_img_size = awim_png.size
        awim_img_center = np.divide(awim_img_size, 2)
        awim_image_object = astroimage.AstroImage(awim_dictionary_text) # create the awim object for the image
        pxs_per_minute = awim_image_object.pxsperdeg_ballpark * 15/60
        awim_png_littleblur = awim_png.filter(filter=BoxBlur(pxs_per_minute*5))
        awim_png_bigblur = awim_png.filter(filter=BoxBlur(pxs_per_minute*17))
        # awim_png_littleblur.save(fp=awim_filename+'littleblur.png') # TODO comment this out
        # awim_png_bigblur.save(fp=awim_filename+'bigblur.png') # TODO comment this out
        # img_alpha_channel = np.array(list(awim_png_bigblur.getdata(band=3))).reshape(awim_png_bigblur.size[::-1])
        
        for key, value in celestial_objs_dictionary.items():
            img_astro_data_array = np.zeros((moments.size, 8)) # empty array of appropriate size. One array per celestial object per image. Images x objects number of these arrays.
            # each row:
            # 0,1 KVpx
            # 2 PSpx y values only. x are same as KVpx
            # 3 object within image bounds True/False
            # 4 opacity on little blurred alpha channel of image
            # 5 opacity on big blurred alpha channel of image
            # 6 px distance from center
            img_astro_data_array[:,[0,1]] = awim_image_object.azalts_to_pxs(value[:,[1,2]], 'KVpx')
            img_astro_data_array[:,2] = np.subtract(img_dims[1], img_astro_data_array[:,[1]]).flatten() # PSpx y
            img_astro_data_array[:,3] = (img_astro_data_array[:,0]>0-sun_moon_size_px[0]/2)&(img_astro_data_array[:,0]<awim_img_size[0]+sun_moon_size_px[0]/2)&(img_astro_data_array[:,1]>0-sun_moon_size_px[1]/2)&(img_astro_data_array[:,1]<awim_img_size[1]+sun_moon_size_px[1]/2)
            index_counter = 0
            for PSpx in img_astro_data_array[:,[0,2]]:
                if (0<PSpx[0]<img_dims[0])&(0<PSpx[1]<img_dims[1]):
                    img_astro_data_array[index_counter,4] = awim_png_littleblur.getpixel(tuple(PSpx))[3]
                    img_astro_data_array[index_counter,5] = awim_png_bigblur.getpixel(tuple(PSpx))[3]
                else:
                    img_astro_data_array[index_counter,4] = 255
                    img_astro_data_array[index_counter,5] = 255
                index_counter += 1
            img_astro_data_array[:,6] = np.sqrt(np.add(np.square(np.subtract(img_astro_data_array[:,0], awim_img_center[0])), np.square(np.subtract(img_astro_data_array[:,1], awim_img_center[1]))))

            imgs_objs_dictionary[awim_filename][key] = img_astro_data_array
            
            track_condition = np.where(img_astro_data_array[:,4] < 255) # indexes of objects on the image not behind little blur opacity
            to_save = np.concatenate((np.array(track_condition[0]).reshape(-1,1), np.full((track_condition[0].size, 1), awim_filename), np.full((track_condition[0].size, 1), key), value[track_condition][:,[0,2]], img_astro_data_array[track_condition][:,4:7]), axis=1)
            tracker_list.append(to_save.reshape(-1).tolist()) # awim filled in

    tracker_list_flattened = [val for sublist in tracker_list for val in sublist]
    tracker_list_oflists = zip(*[iter(tracker_list_flattened)]*8)
    # tracker_array = np.array(tracker_list_flattened).reshape(-1,7) # this works also but the list thing skips making an array
    dictionary_TOC = pd.DataFrame(tracker_list_oflists, columns=['step_count', 'awim', 'object', 'moment', 'altitude', 'little blur opacity', 'big blur opacity', 'px distance from center'])
    dictionary_TOC['step_count'] = dictionary_TOC['step_count'].astype(int)
    dictionary_TOC[['altitude', 'little blur opacity', 'big blur opacity', 'px distance from center']] = dictionary_TOC[['altitude', 'little blur opacity', 'big blur opacity', 'px distance from center']].astype(float)
    dictionary_TOC.to_csv('tracker dataframe.csv') # TODO comment this out

    return imgs_objs_dictionary, dictionary_TOC


# to display the moon partially illuminated I need the angle it appears to be illuminated.
# return degrees. straight down = sun stright below moon = 0°. (+) angle is CCW = illum up the right side. (-) angle is CW = illum up the left side.
# see diagrams for variable meanings.
def calculate_astro_moon_brightsidedirection(moon_azalts_deg, sun_azalts_deg):
    # moon_azalts[:,2] = np.where(np.greater_equal(moon_azalts[:,2], 0), moon_azalts[:,2], 0)
    moon_azalts = moon_azalts_deg * math.pi/180
    sun_azalts = sun_azalts_deg * math.pi/180

    # get abs + direction matrix for moon and sun alts
    # moon_alts_positive = np.where(moon_azalts[:,1] >= 0)
    # sun_alts_positive = np.where(sun_azalts[:,1] >= 0)
    moon_alts_abs = np.abs(moon_azalts[:,1])
    sun_alts_abs = np.abs(sun_azalts[:,1])

    # get the az_rel of the sun relative to the moon, from -180 to +180, then abs + direction matrix
    simple_subtract = np.subtract(sun_azalts[:,0], moon_azalts[:,0])
    big_angle_correction = np.where(simple_subtract > 0, -2*math.pi, 2*math.pi)
    az_rels = np.where(np.abs(simple_subtract) <= math.pi, simple_subtract, np.add(simple_subtract, big_angle_correction))
    degazrels = az_rels * 180/math.pi
    az_rels_direction = np.where(az_rels < 0, -1, 1)
    az_rels_abs = np.abs(az_rels)

    # mh = np.subtract(math.pi/2, np.arcsin(np.multiply(np.cos(moon_alts_abs), np.cos(az_rels_abs))))
    # degmh = mh * 180/math.pi
    # M = np.subtract(math.pi/2, np.arcsin(np.multiply(np.tan(moon_alts_abs), np.tan(math.pi/2 - mh))))
    # degM = M * 180/math.pi
    # A = np.subtract(math.pi/2, np.arcsin(np.multiply(np.tan(az_rels_abs), np.tan(math.pi/2 - mh))))
    # degA = A * 180/math.pi
    sh = np.subtract(math.pi/2, np.arcsin(np.multiply(np.cos(sun_alts_abs), np.cos(az_rels_abs))))
    degsh = sh * 180/math.pi
    # S = np.subtract(math.pi/2, np.arcsin(np.multiply(np.tan(sun_alts_abs), np.tan(math.pi/2 - sh))))
    # degS = S * 180/math.pi
    B = np.subtract(math.pi/2, np.arcsin(np.multiply(np.tan(az_rels_abs), np.tan(math.pi/2 - sh))))
    degB = B * 180/math.pi

    B2 = np.where(sun_azalts[:,1] >= 0, math.pi/2 - B, math.pi/2 + B)
    degB2 = B2 * 180/math.pi
    sm = np.arccos(np.add(np.multiply(np.cos(B2), np.multiply(np.sin(moon_alts_abs), np.sin(sh))), np.multiply(np.cos(moon_alts_abs), np.cos(sh))))
    degsm = sm * 180/math.pi
    moon_brightsidedirections_abs = np.arccos(np.divide(np.subtract(np.cos(sh), np.multiply(np.cos(moon_alts_abs), np.cos(sm))), np.multiply(np.sin(moon_alts_abs), np.sin(sm))))

    moon_brightsidedirections = np.multiply(moon_brightsidedirections_abs, az_rels_direction * 180/math.pi)

    return moon_brightsidedirections


# I don't know why this doesn't return exactly a full/new moon - differs by ~30 minutes? Why? phase is different from "ecliptic longitude different by 180°?" - but since I did it here it is...
# TODO calculate the 180deg ecliptic longitude difference full moon and see if different
def calculate_astro_newfullmoon_andillum(moment_now):
    moment_now_astropy = Time(moment_now)
    moon_illumination_percent = astroplan.moon_illumination(moment_now_astropy) * 100

    # get the nearest full moon, which for astroplan is zero / minimum for some UNEXPLAINED reason.
    check_period = np.timedelta64(30, 'D')
    check_discretization = np.timedelta64(1, 'D')
    moments_check_array = np.arange(moment_now-check_period/2, moment_now+check_period/2, check_discretization)
    moments_check_astropy = Time(moments_check_array)
    moon_phase_check = astroplan.moon_phase_angle(moments_check_astropy).to_value()

    check_period_start = moments_check_array[np.argmin(moon_phase_check)]
    check_period = np.timedelta64(2, 'D') # because I use subtract and add with it later
    check_discretization = np.timedelta64(1, 'h')
    moments_check_array = np.arange(check_period_start-check_period/2, check_period_start+check_period/2, check_discretization)
    moments_check_astropy = Time(moments_check_array)
    moon_phase_check = astroplan.moon_phase_angle(moments_check_astropy).to_value()

    check_period_start = moments_check_array[np.argmin(moon_phase_check)]
    check_period = np.timedelta64(2, 'h') # because I subtract and add later
    check_discretization = np.timedelta64(5, 'm')
    moments_check_array = np.arange(check_period_start-check_period/2, check_period_start+check_period/2, check_discretization)
    moments_check_astropy = Time(moments_check_array)
    moon_phase_check = astroplan.moon_phase_angle(moments_check_astropy).to_value()
    phase_percent = (math.pi - moon_phase_check) / math.pi

    nearest_full_moon = moments_check_array[np.argmin(moon_phase_check)]
    # uhhh. What? The following should not work. Sign of the apocalypse and sign that industrial time is confusing. Coders gave up here. Time objects are a mess.
    # but it does work. Wow.
    # full_moon_datetime = nearest_full_moon.tolist().replace(tzinfo=timezone.utc)

    # get the nearest new moon, which for astroplan is pi / maximum for some UNEXPLAINED reason.
    check_period = np.timedelta64(30, 'D')
    check_discretization = np.timedelta64(1, 'D')
    moments_check_array = np.arange(moment_now-check_period/2, moment_now+check_period/2, check_discretization)
    moments_check_astropy = Time(moments_check_array)
    moon_phase_check = astroplan.moon_phase_angle(moments_check_astropy).to_value()

    check_period_start = moments_check_array[np.argmax(moon_phase_check)]
    check_period = np.timedelta64(2, 'D') # because I use subtract and add with it later
    check_discretization = np.timedelta64(1, 'h')
    moments_check_array = np.arange(check_period_start-check_period/2, check_period_start+check_period/2, check_discretization)
    moments_check_astropy = Time(moments_check_array)
    moon_phase_check = astroplan.moon_phase_angle(moments_check_astropy).to_value()

    check_period_start = moments_check_array[np.argmax(moon_phase_check)]
    check_period = np.timedelta64(2, 'h') # because I subtract and add later
    check_discretization = np.timedelta64(5, 'm')
    moments_check_array = np.arange(check_period_start-check_period/2, check_period_start+check_period/2, check_discretization)
    moments_check_astropy = Time(moments_check_array)
    moon_phase_check = astroplan.moon_phase_angle(moments_check_astropy).to_value()
    phase_percent = (math.pi - moon_phase_check) / math.pi

    nearest_new_moon = moments_check_array[np.argmax(moon_phase_check)]
    # new_moon_datetime = nearest_new_moon.tolist().replace(tzinfo=timezone.utc)

    return nearest_new_moon, nearest_full_moon, moon_illumination_percent


# this is written one moment at a time but could and probably should be written to return a bymoment matrix. Would be a cumbersome matrix though
def get_moon_nearest_and_quarter(moment_now, nearest_new_moon, nearest_full_moon):
    # lunar_cycle_average = np.timedelta64(29.53, 'D') # not used but here it is
    delta_newmoon = abs(moment_now - nearest_new_moon)
    delta_fullmoon = abs(moment_now - nearest_full_moon)
    if (delta_newmoon < delta_fullmoon) and (moment_now <= nearest_new_moon):
        moon_tuple = (np.timedelta64(delta_newmoon, 'D').astype(int), np.timedelta64(delta_newmoon, 'h').astype(int)%24)
        moon_phase_str = '%i days, %i hours until new moon' % moon_tuple
        moon_qtr_day = (4, moon_tuple[0])
        if moon_tuple[0] == 0: # special case return to the start
            moon_day_index = 0
        else:
            moon_day_index = 30-moon_tuple[0] # can be from 30-1=29 to 30-7=23
    elif (delta_newmoon < delta_fullmoon) and (moment_now >= nearest_new_moon):
        moon_tuple = (np.timedelta64(delta_newmoon, 'D').astype(int), np.timedelta64(delta_newmoon, 'h').astype(int)%24)
        moon_phase_str = '%i days, %i hours since new moon' % moon_tuple
        moon_qtr_day = (1, moon_tuple[0])
        moon_day_index = moon_tuple[0]
    elif (delta_newmoon >= delta_fullmoon) and (moment_now <= nearest_full_moon):
        moon_tuple = (np.timedelta64(delta_fullmoon, 'D').astype(int), np.timedelta64(delta_fullmoon, 'h').astype(int)%24)
        moon_phase_str = '%i days, %i hours until full moon' % moon_tuple
        moon_qtr_day = (2, moon_tuple[0])
        moon_day_index = 15-moon_tuple[0] # can be from 15-7=08 to 15-0=15
    elif (delta_newmoon >= delta_fullmoon) and (moment_now >= nearest_full_moon):
        moon_tuple = (np.timedelta64(delta_fullmoon, 'D').astype(int), np.timedelta64(delta_fullmoon, 'h').astype(int)%24)
        moon_phase_str = '%i days, %i hours since full moon' % moon_tuple
        moon_qtr_day = (3, moon_tuple[0])
        moon_day_index = 15+moon_tuple[0] # can be from 15+0=15 to 15+7=22

    return moon_phase_str, moon_day_index, moon_qtr_day


def get_nearest_dailyevents(moment_now, sun_daily, moon_daily):

    sun_time_separation_array = np.subtract(sun_daily, moment_now)

    passed_events_indexes = np.where(np.less_equal(sun_time_separation_array, np.timedelta64(0, 's')))[0]
    passed_events_timesince = np.abs(sun_time_separation_array[passed_events_indexes])
    sun_just_passed_event = passed_events_indexes[np.argmin(passed_events_timesince)]
    sun_timesince = passed_events_timesince[np.argmin(passed_events_timesince)]

    future_events_indexes = np.where(np.greater(sun_time_separation_array, np.timedelta64(0, 's')))[0]
    future_events_timeuntil = np.abs(sun_time_separation_array[future_events_indexes])
    sun_next_event = future_events_indexes[np.argmin(future_events_timeuntil)]
    sun_timeuntil = future_events_timeuntil[np.argmin(future_events_timeuntil)]

    sun_times_tuple = (np.timedelta64(sun_timesince, 'h').astype(int), np.timedelta64(sun_timesince, 'm').astype(int)%60, np.timedelta64(sun_timeuntil, 'h').astype(int), np.timedelta64(sun_timeuntil, 'm').astype(int)%60)

    if any(i == sun_just_passed_event for i in (0, 1, 2)):
        sun_string = '%i:%.2i since sunrise. %i:%.2i until high noon.' % sun_times_tuple
    elif any(i == sun_just_passed_event for i in (3, 4, 5)):
        sun_string = '%i:%.2i since high noon. %i:%.2i until sunset.' % sun_times_tuple
    elif any(i == sun_just_passed_event for i in (6, 7, 8)):
        sun_string = '%i:%.2i since sunset. %i:%.2i until midnight.' % sun_times_tuple
    elif any(i == sun_just_passed_event for i in (9, 10, 11)):
        sun_string = '%i:%.2i since midnight. %i:%.2i until sunrise.' % sun_times_tuple

    moon_time_separation_array = np.abs(np.subtract(moon_daily, moment_now))
    moon_nearest_event = np.argmin(moon_time_separation_array)
    moon_nearest_moment = moon_daily[moon_nearest_event]
    moon_nearest_timedelta = moon_time_separation_array[moon_nearest_event]
    moon_times_tuple = (np.timedelta64(moon_nearest_timedelta, 'h').astype(int), np.timedelta64(moon_nearest_timedelta, 'm').astype(int)%60)

    if any(i == moon_nearest_event for i in (0, 1, 2)) and np.less(moment_now, moon_nearest_moment):
        moon_string = '%i:%.2i until moonrise.' % moon_times_tuple
    elif any(i == moon_nearest_event for i in (0, 1, 2)) and np.greater_equal(moment_now, moon_nearest_moment):
        moon_string = '%i:%.2i since moonrise.' % moon_times_tuple
    elif any(i == moon_nearest_event for i in (3, 4, 5)) and np.less(moment_now, moon_nearest_moment):
        moon_string = '%i:%.2i until moonset.' % moon_times_tuple
    elif any(i == moon_nearest_event for i in (3, 4, 5)) and np.greater_equal(moment_now, moon_nearest_moment):
        moon_string = '%i:%.2i since moonset.' % moon_times_tuple

    return sun_string, moon_string


def awim_chooser(animation_type, moments, celestial_objs_dictionary, imgs_objs_dictionary, TOC_df, placeholder_image):
    first_moment = moments[0]
    last_moment = moments[-1]
    bymoment_awims = np.full(moments.size, False, dtype=list)

    # fill in the awims for sun
    sun_df = TOC_df[TOC_df['object']=='sun']
    # find with medium opacity, meaning near objects in image.
    # find sunrise
    sun_df_sunriseandset = sun_df[(sun_df['altitude'] > -6) & (sun_df['altitude'] < 6)]
    sun_df_sunriseandset.sort_values('px distance from center', ascending=True)
    riseandset_pick = sun_df_sunriseandset['awim'].values[0]
    TOC_indexes = sun_df_sunriseandset.index.tolist()
    for TOC_index in TOC_indexes:
        pick_step = TOC_df.iloc[TOC_index]['step_count']
        if not bymoment_awims[pick_step]:
            bymoment_awims[pick_step] = riseandset_pick

    sun_df_opacityfinder = (sun_df['big blur opacity'] > 0) & (sun_df['big blur opacity'] < 200)
    awims_sun_opacity = sun_df[sun_df_opacityfinder]
    awim_sun_opacity_picks = awims_sun_opacity['awim'].value_counts().index.tolist()
    for opacity_pick in awim_sun_opacity_picks:
        pick_steps = sun_df[sun_df['awim']==opacity_pick]['step_count'].values
        bymoment_awims[pick_steps] = np.where(np.logical_not(bymoment_awims[pick_steps]), opacity_pick, bymoment_awims[pick_steps])

    sun_df_sorted_pxcenterdist = sun_df.sort_values('px distance from center', ascending=True)
    # sun_df.to_csv('sun px dist from center.csv') # TODO comment this out
    sorted_df_indexes = sun_df_sorted_pxcenterdist.index.tolist()
    for TOC_index in sorted_df_indexes:
        pick_step = TOC_df.iloc[TOC_index]['step_count']
        if not bymoment_awims[pick_step]:
            bymoment_awims[pick_step] = TOC_df.iloc[TOC_index]['awim']
    
    # fill awims for moon
    moon_df = TOC_df[TOC_df['object']=='moon']
    # TODO make the list only unique values
    moon_df.sort_values('px distance from center', ascending=True, inplace=True)
    moon_df.to_csv('moon picks dataframe.csv') # TODO comment this out
    sorted_df_indexes = moon_df.index.tolist()
    for TOC_index in sorted_df_indexes:
        pick_step = TOC_df.iloc[TOC_index]['step_count']
        if not bymoment_awims[pick_step]:
            bymoment_awims[pick_step] = TOC_df.iloc[TOC_index]['awim']

    bymoment_awims[:] = np.where(np.logical_not(bymoment_awims), placeholder_image, bymoment_awims)

    awims_with_sun = pd.unique(sun_df['awim'])
    
    return bymoment_awims


# converts center position of image to bottom left coordinate of image to place image
# TODO orient the image based on desired angle
# TODO size the image based on multiple desired coords in image 
def img_placer(pos_want, img_dims, obj_dims):
    x_poshint = float((pos_want[0]-obj_dims[0]/2) / img_dims[0]) # float required to convert to regular python float instead of numpy float64
    y_poshint = float((pos_want[1]-obj_dims[1]/2) / img_dims[1]) # float required to convert to regular python float instead of numpy float64
    if not ((0 <= x_poshint <= 1) and (0 <= y_poshint <= 1)):
        x_poshint = 1.0
        y_poshint = 1.0
    pos_dictionary = {'x':x_poshint, 'y':y_poshint}
    # pos = (pos_want[0]-img_size[0]/2, pos_want[1]-img_size[1]/2)

    return pos_dictionary