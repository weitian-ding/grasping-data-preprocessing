import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# parse the daily origin recording from Google sheets and produce an interpolation of daily origin given dates
def parse_daily_origin(filepath):
    daily_origin = pd.read_csv(filepath, usecols=['Date', 'Time', 'X-origin', 'y-origin', 'z-origin'])
    daily_origin['Date'] = pd.to_datetime(daily_origin['Date'])
    daily_origin['Time'] = daily_origin['Time'].fillna('AM')  # assumes missing time is AM

    # extract timestamp from date and time
    def _parse_timestamp(r):
        date = r['Date']
        time = r['Time']

        # if the time is AM or PM, then 0 hours and 12 hours are added to date respectively
        if time.lower() == 'AM':
            return date

        if time.lower() == 'PM':
            return date + pd.Timedelta('12 hours')

        # try to parse time string
        try:
            time = pd.Timedelta(time)
            return date + time
        except ValueError:
            return date

    daily_origin['timestamp'] = daily_origin.apply(_parse_timestamp, axis=1)

    # convert timestamp to nanoseconds
    daily_origin['ts_value'] = daily_origin['timestamp'].map(lambda ts: ts.value)

    daily_origin_xyz_lookups = []
    for axis in ['X-origin', 'y-origin', 'z-origin']:
        # construct x (timestamp in nanoseoncs) and y (daily origin axis i values) for linear interpolation
        days_origin_axis_i = daily_origin[['ts_value', axis]].dropna()

        # computes interpolation, f(days) => axis value
        axis_lookup = _get_timestamp_value_lookup(days_origin_axis_i['ts_value'], days_origin_axis_i[axis])

        daily_origin_xyz_lookups.append(axis_lookup)

    return DailyOrigin(daily_origin_xyz_lookups)


# prduces a timestamp => value look up
# given a novel timestamp, the lookup retrieve the value associated with the previous timestamp in the lookup
def _get_timestamp_value_lookup(timestamps, values):
    # given a timestamp for grasp session, 'previous' interpolation will find the closest previous daily origin,
    # which is desired under the assumption that daily origin recording always take place before grasp sessions
    lookup = interp1d(timestamps, values, fill_value='extrapolate', kind='previous')

    return lookup


# a wrapper for getting (x, y, z) daily origin given date
class DailyOrigin(object):

    def __init__(self, daily_origin_xyz_lookups):
        self.daily_origin_x_lookup = daily_origin_xyz_lookups[0]
        self.daily_origin_y_lookup = daily_origin_xyz_lookups[1]
        self.daily_origin_z_lookup = daily_origin_xyz_lookups[2]

    def get_origin_by_session_timestamp(self, timestamp):
        """ Computes (x, y, z) given date

        :param timestamp: a pandas Datetime object
        :return: a list [x, y, z] representing the origin
        """

        origin_x = self.daily_origin_x_lookup(timestamp.value)
        origin_y = self.daily_origin_y_lookup(timestamp.value)
        origin_z = self.daily_origin_z_lookup(timestamp.value)

        return (origin_x, origin_y, origin_z)
