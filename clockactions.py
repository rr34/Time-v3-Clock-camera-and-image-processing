import numpy as np
import clockmath, formatters

def get_time_strings(location, elevation_msl, currenttime, nowmoments):
    currenttime = np.datetime64(currenttime)
    nowmoments = formatters.format_datetime(nowmoments, direction='from list of ISO 8601 strings')
    nowmoments = np.array(nowmoments).astype('datetime64[ns]')
    sundaily, moondaily = clockmath.calculate_astro_risesandsets(location, currenttime, elevation_msl) # todo: cache these results because they take time to calculate.
    daynightlengths = clockmath.calculate_astro_daynightlength(nowmoments, sundaily)
    print('stophere')
    # todo: nearest sunrise sunset noon midnight strings, nearest moon phase event strings, nearest moonrise and moonset strings