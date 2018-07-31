import logging

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


# parse the daily origin recording from Google sheets and produce an interpolation of daily origin given dates
def parse_daily_origin(filepath):
    daily_origin = pd.read_csv(filepath, usecols=['Date', 'X-origin', 'y-origin', 'z-origin'])
    daily_origin['Date'] = pd.to_datetime(daily_origin['Date'])

    # drop duplicated recordings for a same date
    daily_origin = daily_origin.drop_duplicates(subset=['Date'])

    # sort by date so that daily_origin['Date'][0] is the first date of recording
    daily_origin = daily_origin.sort_values(by='Date')

    date_begin = daily_origin['Date'][0]

    # computes days relative to date_begin
    daily_origin['days'] = daily_origin['Date'].map(lambda d: (d - date_begin).days)

    daily_origin_xyz_interps = []
    for axis in ['X-origin', 'y-origin', 'z-origin']:
        # prepare x and y for interpolation
        days_origin_axis_i = daily_origin[['days', axis]].dropna()

        # computes interpolation, f(days) => axis value
        axis_interp = _records_to_linear_interp(days_origin_axis_i['days'], days_origin_axis_i[axis])

        daily_origin_xyz_interps.append(axis_interp)

    return DailyOrigin(date_begin, daily_origin_xyz_interps)


# fill nans at the head and tail of a list
def _records_to_linear_interp(days, values):
    interp = interp1d(days, values, fill_value='extrapolate', kind='linear')

    return interp


# a wrapper for getting (x, y, z) daily origin given date
class DailyOrigin(object):

    def __init__(self, date_begin, daily_origin_interps):
        self.date_begin = date_begin
        self.daily_origin_xyz_interps = daily_origin_interps

    def get_origin_by_date(self, date):
        """ Computes (x, y, z) given date

        :param date: a pandas Datetime object
        :return: a list [x, y, z] representing the origin
        """

        days = (date - self.date_begin).days

        if days < 0:
            logging.warning('%s is earlier than %s, extrapolation is used in daily origin', date, self.date_begin)

        origin = [np.asscalar(f(days)) for f in self.daily_origin_xyz_interps]

        return origin