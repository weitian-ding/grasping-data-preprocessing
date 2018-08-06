import logging
from collections import OrderedDict

import pandas as pd
# parse the daily origin recording from Google sheets and produce an interpolation of daily origin given dates
from tqdm import tqdm


def parse_daily_origin(filepath):
    daily_origin_df = pd.read_csv(filepath, usecols=['Date', 'Time', 'X-origin', 'y-origin', 'z-origin'])
    daily_origin_df['Date'] = pd.to_datetime(daily_origin_df['Date'])
    daily_origin_df['Time'] = daily_origin_df['Time'].fillna('AM')  # assumes missing time is AM

    # extract a timestamp for each measurement of origin
    def _extract_measurement_timestamp(r):
        date = r['Date']
        time = r['Time']

        # if the time is AM or PM, then 0 hours and 12 hours are added to date respectively
        if time.lower() == 'am':
            return date

        if time.lower() == 'pm':
            return date + pd.Timedelta('12 hours')

        # try to parse time string
        try:
            time = pd.Timedelta(time)
            return date + time
        except ValueError:
            # fail to parse the time string
            return date

    daily_origin_df['timestamp'] = daily_origin_df.apply(_extract_measurement_timestamp, axis=1)
    daily_origin_ordered_lookup = _build_daily_origin_lookup(daily_origin_df=daily_origin_df)

    return DailyOriginLookup(daily_origin_ordered_lookup)


def _build_daily_origin_lookup(daily_origin_df):
    """Produces a (session timestamp => list of of origin values) lookup
    there can be multiple sessions during the day, the origin values measured for the same date is ordered by
    when the origin is measured during the day

    :param daily_origin_df: the dataframe for daily origin recordings
    :return: an ordered dict of daily origin measurement lookups
    """
    daily_origin_df = daily_origin_df.dropna(subset=['X-origin', 'y-origin', 'z-origin'])
    # sort daily origin df by session timestamp
    daily_origin_df = daily_origin_df.sort_values(by=['timestamp'])
    daily_origin_lookup = OrderedDict()

    print('building daily origin lookup table...')
    for r in tqdm(daily_origin_df.to_dict(orient='records')):
        session_date_key = r['Date']
        origin = (r['X-origin'], r['y-origin'], r['z-origin'])
        if session_date_key in daily_origin_lookup:
            daily_origin_lookup[session_date_key].append((origin))
        else:
            daily_origin_lookup[session_date_key] = [origin]

    return daily_origin_lookup


# a wrapper for getting (x, y, z) daily origin given date
class DailyOriginLookup(object):

    def __init__(self, daily_origin_ordered_lookup):
        self._daily_origin_ordered_lookup = daily_origin_ordered_lookup

    def lookup_origin_by_session_date_and_id(self, session_date, session_id):
        """ lookup daily origin, (x, y, z), given session date and session id
        if no origin measurement is taken for date, then a measurement for a subsequent date will be returned

        :param session_date: the date of a grasp session
        :param session_id: the integer index of a session during the day, at the moment session id can only be 0 and 1
        :return: a tuple (x, y, z) representing the origin
        """

        for d, m in self._daily_origin_ordered_lookup.items():
            if d >= session_date:
                if d > session_date:
                    logging.warning('no origin measurement for date {}, '
                                    'using the first measurement for date {}'.format(session_date, d))
                    return m[0]

                # try to retrieve the measurement for session id
                try:
                    measurement = m[session_id]
                except IndexError:
                    logging.warning('no origin measurement for session {} at date {}, '
                                    'using the first measurement taken at that date'.format(session_id, session_date))
                    measurement = m[0] # fall back to the first measurement at date d

                return measurement

        raise ValueError('no suitable origin measurement found for '
                         'date {} and session id {}'.format(session_date, session_id))
