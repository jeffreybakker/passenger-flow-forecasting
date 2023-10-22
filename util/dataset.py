import glob
import math
from datetime import datetime, timedelta, date
from typing import Optional, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from util.event import EVENT_TYPE_MAP
from util.graph import read_stations


class PassengerFlowDataset(Dataset):
    _data: pd.DataFrame

    _window: int
    _horizon: int

    _index_offset: int

    _transform: Optional[Callable]

    def __init__(self,
                 window: int = 168,
                 horizon: int = 72,
                 origin: Optional[str] = None,
                 destination: Optional[str] = None,
                 min_date: Optional[date] = None,
                 max_date: Optional[date] = None,
                 transform: Optional[Callable] = None):
        """
        Loads passenger flow data from disk. Expects CSV files with columns ['date', 'hour', 'origin', 'destination',
        'passengers'].
        :param window: the amount of historic observations to include
        :param horizon: the amount of future observations to return as target
        :param min_date: if set ignores all records from before this date
        :param max_date: if set ignores all records from after this date
        :param transform: transformation to apply to `passengers` column
        """
        self._window = window
        self._horizon = horizon
        self._transform = transform

        # Load data
        progress = tqdm(desc='LOADING DATA', total=6)
        self._data = self._load_data(origin, destination, min_date, max_date, progress)
        progress.close()

        # Compute the index offset
        self._index_offset = self._data.loc[:self._data.index.levels[0].min() + timedelta(hours=self._window - 1), :, :] \
            .shape[0]

    def __len__(self) -> int:
        """
        Returns the amount of valid observations in the dataset
        :return: the amount of valid observations >= 0
        """
        start = self._data.index.levels[0].min() + timedelta(hours=self._window)
        end = self._data.index.levels[0].max() - timedelta(hours=self._horizon)
        return self._data.loc[start:end, :, :].index.size

    def _get_data(self, idx: int):
        dt, origin, destination = self._data.iloc[self._index_offset + idx].name

        window = self._data.loc[dt - timedelta(hours=self._window):dt, origin, destination]
        horizon = self._data.loc[dt:dt + timedelta(hours=self._horizon), origin, destination]

        return window, horizon

    def __getitem__(self, idx: int):
        window, horizon = self._get_data(idx)

        if self._transform:
            window = self._transform(window)
            horizon = self._transform(horizon)

        return window, horizon

    def _load_data(self,
                   origin: Optional[str],
                   destination: Optional[str],
                   min_date: Optional[date],
                   max_date: Optional[date],
                   progress: tqdm) -> pd.DataFrame:
        """
        Loads the CSV files from disk and prepares the DataFrame for Time Series applications.
        :param min_date: if set ignores all records from before this date (inclusive)
        :param max_date: if set ignores all records from after this date (exclusive)
        :return: a DataFrame with columns ['passengers'], indexed by ['datetime', 'origin', 'destination']
        """
        # Load all files in the root directory
        try:
            df = pd.concat([
                pd.read_csv(f)
                for f in map(lambda s: s.replace('\\', '/'), glob.glob('data/flow/**/*.csv', recursive=True))
            ])[['date', 'hour', 'origin', 'destination', 'passengers']]
        except ValueError:
            df = pd.concat([
                pd.read_csv(f)
                for f in map(lambda s: s.replace('\\', '/'), glob.glob('../data/flow/**/*.csv', recursive=True))
            ])[['date', 'hour', 'origin', 'destination', 'passengers']]
        progress.update()

        # Select queried time range
        if min_date:
            df = df[df.date >= min_date.strftime('%Y-%m-%d')]
        if max_date:
            df = df[df.date < max_date.strftime('%Y-%m-%d')]

        # Select the queried origin and destination stations
        if origin:
            df = df[df.origin == origin]
        if destination:
            df = df[df.destination == destination]

        progress.update()

        # Parse the date and hour columns to a python `datetime` object
        df['datetime'] = df.apply(
            lambda row: datetime.strptime(f'{row.date} {row.hour:02d}:00:00', '%Y-%m-%d %H:%M:%S'),
            axis=1)
        progress.update()

        # Select useful columns and set index
        df = df[['datetime', 'origin', 'destination', 'passengers']] \
            .reset_index(drop=True) \
            .set_index(['datetime', 'origin', 'destination']) \
            .sort_index()

        # Reindex in order to fill the missing gaps (important for Time Series Analysis or Forecasting)
        start_date, end_date = df.index.levels[0].min(), df.index.levels[0].max()
        date_range = end_date - start_date
        index = pd.MultiIndex.from_tuples([
            (start_date + timedelta(hours=h), o, d)
            for o, d in df.index.droplevel(0).unique()
            for h in range(0, date_range.days * 24 + date_range.seconds // 3600 + 1)],
            names=['datetime', 'origin', 'destination'])
        df = df.reindex(index).sort_index()
        progress.update()

        # Fill empty values (NaN) with 0.0
        df.fillna(0.0, inplace=True)
        progress.update()

        # Insert error value needed for Moving Average (MA) model as part of SARIMA
        df = df.assign(noise=np.random.normal(size=df.shape[0]))
        progress.update()

        return df


class FeaturePassengerFlowDataset(PassengerFlowDataset):
    _augmented: pd.DataFrame
    _augment_idx: int

    def __init__(self,
                 window: int = 168,
                 horizon: int = 72,
                 origin: Optional[str] = None,
                 destination: Optional[str] = None,
                 min_date: Optional[date] = None,
                 max_date: Optional[date] = None,
                 transform: Optional[Callable] = None,
                 augment_events: Optional[float] = None):
        super().__init__(window, horizon, origin, destination, min_date, max_date, transform)
        self._augmented = self._augment_data(augment_events)

    def _augment_data(self, augment_events: Optional[float]) -> pd.DataFrame:
        # Start augmented data after all other data
        self._augment_idx = super().__len__()

        if augment_events is None or augment_events < 1.0:
            return self._data.head(0).copy()

        # Generate a new column with the observation index being referred to
        df = self._data \
            .reset_index(drop=False) \
            .reset_index(drop=False) \
            .rename(columns={'index': 'idx'})
        df.idx = df.idx - self._index_offset

        # Drop non-event observations or observations that are out of scope
        df = df[df.event_capacity > 0].copy()
        df = df[(df.idx >= 0) & (df.idx <= self._augment_idx)]

        # Sample X% events
        df = df.sample(frac=augment_events - 1.0, replace=True)

        return df

    def __len__(self):
        return super().__len__() + self._augmented.shape[0]

    def _get_data(self, idx: int):
        if idx >= self._augment_idx:
            augmented = self._augmented.iloc[idx - self._augment_idx]
            window, horizon = super()._get_data(augmented.idx)

            # augmented
            window.iloc[-1] = augmented.iloc[-window.shape[1]:]
            return window, horizon
        else:
            return super()._get_data(idx)

    def _load_data(self, origin: Optional[str], destination: Optional[str],
                   min_date: Optional[date], max_date: Optional[date], progress: tqdm) -> pd.DataFrame:
        progress.reset(progress.total + 3 + 5)

        df = super()._load_data(origin, destination, min_date, max_date, progress)
        df = self._load_date(df, progress)
        # df = self._load_weather(df, progress)
        df = self._load_events(df, min_date, max_date, progress)

        return df

    def _load_date(self, df: pd.DataFrame, progress: tqdm) -> pd.DataFrame:
        # Time of Year and its inverse (1-x)
        # df['timeofyear'] = df.index.to_series() \
        #     .map(lambda idx: (idx[0] - datetime(idx[0].year, 1, 1)).days /
        #                      (datetime(idx[0].year + 1, 1, 1) - datetime(idx[0].year, 1, 1)).days)
        # df['timeofyearinverse'] = df.timeofyear.map(lambda toy: 1 - toy)

        progress.update()

        # Day of the week
        df = df.assign(weekend=df.index.to_series().map(lambda idx: idx[0].weekday() > 4))

        progress.update()

        # Hour of the day
        df = df.assign(**{
            f'hours_{timeslot*4:02d}_{(timeslot+1)*4:02d}': df.index.to_series()
                       .map(lambda idx: idx[0].hour // 4 == timeslot)
            for timeslot in range(6)
        })

        progress.update()

        return df

    def _load_events(self, df: pd.DataFrame,
                     min_date: Optional[date], max_date: Optional[date],
                     progress: tqdm) -> pd.DataFrame:
        # Load event data
        try:
            events = pd.read_csv('data/events.csv')
            venues = pd.read_csv('data/venue_capacity.csv', delimiter=';').set_index('venue')
        except FileNotFoundError:
            events = pd.read_csv('../data/events.csv')
            venues = pd.read_csv('../data/venue_capacity.csv', delimiter=';').set_index('venue')

        stations = read_stations().set_index('codes')
        progress.update()

        # event_types = set(filter(lambda x: x is not None, EVENT_TYPE_MAP.values()))
        event_types = sorted(set(filter(lambda x: x is not None, EVENT_TYPE_MAP.values())))
        events.event_type = events.event_type.map(lambda et: EVENT_TYPE_MAP[et] if et in EVENT_TYPE_MAP else None)
        events = events.dropna(subset=['event_type'])

        # Parse columns and drop invalid values
        events['datetime'] = events.start_date.map(lambda dt: "-".join(dt.split("-")[:-1]).replace("T", " "))
        events['datetime'] = events.datetime.map(
            lambda dt: None if dt == '' else datetime.strptime(dt, '%Y-%m-%d %H:%M'))
        events = events[events.datetime != None][
            ['datetime', 'name', 'event_type', 'venue', 'latitude', 'longitude']].dropna()
        events['datetime'] = events.datetime.map(lambda dt: datetime(dt.year, dt.month, dt.day, dt.hour))

        if min_date:
            min_date = datetime(min_date.year, min_date.month, min_date.day)
            events = events[events.datetime >= min_date]
        if max_date:
            max_date = datetime(max_date.year, max_date.month, max_date.day)
            events = events[events.datetime < max_date]

        progress.update()

        # Match venue to their capacity
        def parse_float(f):
            try:
                return float(f)
            except ValueError:
                return None
            except TypeError:
                return None

        events['capacity'] = events['venue'].map(lambda venue: venues.loc[venue].capacity if venue in venues.index else None)
        events['capacity'] = events.capacity.map(parse_float)

        progress.update()

        # Map to nearest metro station
        events['station'] = events.apply(
            lambda event: next(stations.apply(lambda station: math.sqrt(
                math.pow(event.latitude - station.lat, 2) + math.pow(event.longitude - station.long, 2)) * 6371e3 / 360,
                                              axis=1)
                               .sort_values().items()),
            axis=1)
        events['distance'] = events.station.apply(lambda station: station[1])
        events['station'] = events.station.apply(lambda station: station[0])

        events = events[(events.distance < 500) & (events.capacity > 1100)]
        events = events.sort_values('distance', ascending=False)

        # TODO: TEST LOG VALUES
        events.distance = events.distance / events.distance.max()
        events.capacity = events.capacity / events.capacity.max()

        events = events \
            .groupby(['datetime', 'station']) \
            .apply(lambda df: df.sort_values('capacity', ascending=False).iloc[0])
        events = events.set_index(['datetime', 'station']).sort_index()

        progress.update()

        event_map = df.index.to_series() \
            .apply(lambda idx: (idx[0], idx[2])) \
            .apply(lambda idx: events.loc[idx] if idx in events.index else None)

        df = df.assign(
            event_capacity=event_map.dropna().apply(lambda event: event['capacity']),
        # df.index.to_series().apply(
        #         lambda idx: events.loc[idx[0], idx[2]].capacity if (idx[0], idx[2]) in events.index else 0),
            event_distance=event_map.dropna().apply(lambda event: event['distance']),
            # df.index.to_series().apply(
            #     lambda idx: events.loc[idx[0], idx[2]].distance if (idx[0], idx[2]) in events.index else 0),
            # event_distance_inverse=df.index.to_series().apply(
            #     lambda idx: 1.0 / (events.loc[idx[0], idx[2]].distance + 1e-14)
            #     if (idx[0], idx[2]) in events.index else 0),
            **{f'event_type_{t}': event_map.dropna().apply(lambda event: event['event_type'] == t)
                # df.index.to_series().apply(
                #     lambda idx: events.loc[idx[0], idx[2]].event_type == t if (idx[0], idx[2]) in events.index else False)
               for t in event_types})

        df.event_capacity.fillna(0.0, inplace=True)
        df.event_distance.fillna(0.0, inplace=True)
        for t in event_types:
            df[f'event_type_{t}'].fillna(False, inplace=True)

        progress.update()

        return df

    def _load_weather(self, df: pd.DataFrame, progress: tqdm) -> pd.DataFrame:
        # Load weather data
        weather = pd.read_csv('data/weather.csv')

        progress.update()

        # Parse the date and hour columns to a python `datetime` object
        weather['datetime'] = weather.apply(
            lambda row: datetime.strptime(f'{row.date} {row.hour:02d}:00:00', '%Y-%m-%d %H:%M:%S'),
            axis=1)

        # Boolean whether the weather is clear or not (ie. no rain, snow, thunder, et cetera)
        weather['weather_clear'] = weather.condition.apply(lambda c: 1 if len(c) == 0 else 0)

        # Select useful columns and set index
        weather = weather[['weather_clear', 'temperature', 'windspeed', 'station', 'datetime']] \
            .rename(columns={'station': 'destination'}) \
            .set_index(['datetime', 'destination']) \
            .sort_index()

        progress.update()

        # Update data with weather data
        return df.assign(**{
            col: df.index.to_series().map(lambda idx: weather[col].loc[idx[0], idx[2]]
            if (idx[0], idx[2]) in weather.index else None)
            for col in ['temperature', 'windspeed', 'weather_clear']
        })


class GraphPassengerFlowDataset(FeaturePassengerFlowDataset):

    def __init__(self,
                 window: int = 168,
                 horizon: int = 72,
                 origin: Optional[str] = None,
                 destination: Optional[str] = None,
                 min_date: Optional[date] = None,
                 max_date: Optional[date] = None,
                 transform: Optional[Callable] = None,
                 augment_events: Optional[float] = None):
        super().__init__(window, horizon, origin, destination, min_date, max_date, transform, augment_events)

    def _augment_data(self, augment_events: Optional[float]) -> pd.DataFrame:
        self._augment_idx = self._data.index.levels[0].size - self._window - self._horizon
        return super()._augment_data(augment_events)

    def __len__(self):
        return self._data.index.levels[0].size - self._window - self._horizon + self._augmented.shape[0]

    def _get_data(self, idx: int):
        augment = None
        if idx >= self._augment_idx:
            augment = self._augmented.iloc[idx - self._augment_idx]
            idx = augment.idx

        dt = self._data.index.levels[0][idx + self._window]
        window = self._data.loc[dt - timedelta(hours=self._window):dt, :, :]
        horizon = self._data.loc[dt:dt + timedelta(hours=self._horizon), :, :]

        if augment:
            window.loc[datetime, augment.origin, augment.destination] = augment.iloc[1:]

        return window, horizon
