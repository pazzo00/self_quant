import pandas as pd
import numpy as np
from warnings import warn
import bisect
from base_data_loader import BaseDataLoader
from datetime import timedelta


class ReportDataLoader(BaseDataLoader):
    def __init__(self, data, codes, cols, trade_days,  season_data):
        super().__init__(data, codes, cols, trade_days)
        self.season_data = season_data

    @classmethod
    def load_data(cls, path, start=None, end=None, codes=None, fields=None, lag=None, trade_days=None):
        if (not isinstance(lag, str) or ('Y' not in lag.upper() and 'Q' not in lag.upper()) or
                lag.upper().count('Y') + lag.upper().count('Q') > 1):
            lag = 0
            mode = 'Q'
            day_lag = (lag + 1) * 184
            warn('输入的lag非法或未输入，lag按0Q处理')
        elif 'Y' in lag:
            lag = int(lag[:lag.index('Y')])
            mode = 'Y'
            day_lag = (lag + 1) * 366
        elif 'Q' in lag:
            lag = int(lag[:lag.index('Q')])
            mode = 'Q'
            day_lag = (lag + 1) * 184

        filters = cls.get_time_range(start, end, day_lag)

        if codes is not None:
            filters.append(('code', 'in', codes))
        if fields is not None:
            data = cls.__load_data__(path, ['datetime', 'code', 'report_period'] + fields, filters)
        else:
            data = cls.__load_data__(path, None, filters)
            data = data.loc[:, ~data.columns.str.contains('_id')]
            fields = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            data = data[['datetime', 'code', 'report_period'] + fields]
        data['datetime'] = data['datetime'] + timedelta(hours=15)
        return cls.from_report_dataframe(data, lag, mode, start, end, fields, trade_days, codes)

    @staticmethod
    def get_newest_data(data, val_name, day=None):
        tmp_data = data
        tmp_data = pd.pivot(tmp_data, index='datetime', columns='code', values=val_name)
        tmp_data.sort_index(inplace=True)
        tmp_data = tmp_data.ffill()
        if day is not None:
            tmp_data = tmp_data[tmp_data.index <= pd.to_datetime(day) + timedelta(hours=15)]
            return tmp_data.tail(1)
        return tmp_data

    @classmethod
    def from_report_dataframe(cls, data, lag, mode, start=None, end=None, fields=None, trade_days=None, codes=None):
        if codes is None:
            codes = sorted(data['code'].unique().tolist())

        filters = cls.get_time_range(start, end)
        if trade_days is None:
            if not filters:
                trade_days = pd.read_parquet('../../数据/stock_bar_1day.parquet',
                                             columns=['datetime'], engine='pyarrow')
            else:
                trade_days = pd.read_parquet('../../数据/stock_bar_1day.parquet',
                                             columns=['datetime'], engine='pyarrow', filters=filters)
            trade_days = sorted(trade_days['datetime'].unique().tolist())

        season_data = None
        if mode == 'Y':
            data['month'] = data['report_period'].str[5:7]
            data = data[(data['month'] == '12') & (data['report_period'].str[8:] == '31')]
        else:
            season_data = cls.get_last_season(data, trade_days, codes)

        report_data_dict = {}
        for report_period in sorted(data['report_period'].unique()):
            report_data_dict[report_period] = data[data['report_period'] == report_period]
        report_periods = list(report_data_dict.keys())

        idx = cls.get_season_id(trade_days, codes, report_periods, report_data_dict)
        trade_days = pd.to_datetime(trade_days)

        data, cols = cls.get_lag_data(idx, lag, mode, fields, trade_days, report_periods, report_data_dict, codes)
        return cls(data, codes, cols, trade_days, season_data)

    @staticmethod
    def get_last_season(data, trade_days, codes):
        season_data = data[['datetime', 'code', 'report_period']]
        season_data['season'] = np.int64(data['report_period'].str[5:7]) // 4 + 1
        season_data.sort_values(by=['datetime', 'code', 'report_period'], ascending=True, inplace=True)
        season_data.drop_duplicates(subset=['datetime', 'code'], keep='last', inplace=True)
        season_data = pd.pivot(season_data, index='datetime', columns='code', values='season')
        season_data = season_data.reindex(columns=codes, fill_value=np.nan)
        season_data = season_data.ffill()
        last_data = season_data[season_data.index <= trade_days[0]].tail(1)
        if len(last_data) == 0:
            return season_data.reindex(trade_days).ffill()
        season_data = pd.concat([last_data, season_data.reindex(trade_days)])
        return season_data.ffill().iloc[1:]

    @classmethod
    def get_season_id(cls, trade_days, codes, report_periods, report_data_dict):
        idx = []
        for day in trade_days:
            if not isinstance(day, str): day = day.strftime('%Y-%m-%d')
            tmp_idx = bisect.bisect_right(report_periods, day) - 1
            tmp_data = cls.get_newest_data(report_data_dict[report_periods[tmp_idx]],
                                                        'report_period', day)
            tmp_idx = np.array([tmp_idx] * len(codes))
            if len(tmp_data) == 0:
                tmp_idx = tmp_idx - 1
            else:
                tmp_data = tmp_data.iloc[0].reindex(codes)
                tmp_idx = tmp_idx - 1 * (tmp_data.isna()).values
            idx.append(tmp_idx)
        return np.array(idx) + 1


    @classmethod
    def get_lag_data(cls, idx, lag, mode, fields, trade_days, report_periods, report_data_dict, codes):
        data, cols = [], []
        for lag in range(lag + 1):
            idx -= 1
            unique_idx = np.unique(idx)
            for col in fields:
                lag_data = []
                for unique_i in unique_idx:
                    tmp_data = cls.get_newest_data(report_data_dict[report_periods[unique_i]], col)
                    if tmp_data.iloc[[-1]].index <= trade_days[0]:
                        tmp_data = tmp_data.iloc[[-1]]
                    else:
                        last_data = tmp_data[tmp_data.index < trade_days[0]].tail(1)
                        tmp_data = pd.concat([last_data, tmp_data.reindex(trade_days)]).ffill()
                        if len(last_data):
                            tmp_data = tmp_data.iloc[1:]
                    tmp_data = tmp_data.reindex(columns=codes, fill_value=np.nan)
                    flag = idx == unique_i
                    flag = np.where(flag, 1.0, np.nan)
                    if len(lag_data) == 0:
                        lag_data = flag * tmp_data.values
                    else:
                        lag_data = np.nanmean([lag_data, flag * tmp_data.values], axis=0)
                data.append(lag_data)
                cols.append(col + '_lag{}'.format(lag) + mode)
        data = np.transpose(np.array(data), (1, 0, 2))
        cols = {k: v for v, k in enumerate(cols)}
        return data, cols

