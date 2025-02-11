import pandas as pd
from datetime import datetime, timedelta
from warnings import warn
import bisect


class BaseDataLoader:
    def __init__(self, data, codes, cols, trade_days):
        self.data = data
        self.codes = codes
        self.cols = cols
        self.trade_days = trade_days

    @classmethod
    def load_data(cls, path, start=None, end=None, codes=None, fields=None, lag=None):
        if not isinstance(lag, int):
            lag = 0
            warn('输入的lag非法或未输入，lag按0处理')

        filters = cls.get_time_range(start, end, lag)

        if codes is not None:
            filters.append(('code', 'in', codes))
        if fields is None:
            data = cls.__load_data__(path, None, filters)
        else:
            data = cls.__load_data__(path, ['datetime', 'code'] + fields, filters)

        return cls.from_dataframe(data, codes, fields)

    @staticmethod
    def __load_data__(path, fields, filters):
        if fields is not None:
            if len(filters):
                data = pd.read_parquet(path, engine='pyarrow', columns=fields, filters=filters)
            else:
                data = pd.read_parquet(path, engine='pyarrow', columns=fields)
        else:
            warn('fields未输入，读取数据可能过大')
            if len(filters):
                data = pd.read_parquet(path, engine='pyarrow', filters=filters)
            else:
                data = pd.read_parquet(path, engine='pyarrow')
        return data

    @classmethod
    def from_dataframe(cls, data, codes=None, fields=None):
        cols = [_ for _ in data.columns if _ not in ['datetime', 'code']]
        data = data.pivot(index='datetime', columns='code', values=cols)

        if codes is None:
            codes = data.columns.levels[1]
        if fields is None:
            fields = data.columns.levels[0]
        raw_columns = pd.MultiIndex.from_product([fields, codes])
        data = data.reindex(raw_columns, axis='columns')
        trade_days = data.index.tolist()
        data = data.values.reshape(-1, len(fields), len(codes))
        cols = {k: v for v, k in enumerate(fields)}
        return cls(data, codes, cols, trade_days)

    @staticmethod
    def get_time_range(start, end, lag=0):
        filters = []
        if start is not None:
            start = pd.to_datetime(start) - timedelta(days=lag)
            filters.append(('datetime', '>=', start))
        if end is not None:
            end = pd.to_datetime(end)
            filters.append(('datetime', '<=', end))
        return filters

    def get_window_df(self, field, window, day):
        end = bisect.bisect_right(self.trade_days, pd.to_datetime(day))
        start = max(0, end-window-1)
        return pd.DataFrame(index=self.trade_days[start:end],
                            columns=self.codes, data=self.data[start:end, self.cols[field], :])

    def get_window_df_by_idx(self, field, window, idx):
        start = idx - window
        if start < 0:
            start = 0
            warn('idx小于window，从索引0开始取数')
        return pd.DataFrame(index=self.trade_days[start:idx], columns=self.codes,
                            data=self.data[start:idx, self.cols[field], :])

    def to_dataframes(self):
        return {k: pd.DataFrame(data=self.data[:, v, :],
                                    index=self.trade_days, columns=self.codes) for k, v in self.cols.items()}

    def to_dataframe(self, field=None):
        if field is None:
            return pd.DataFrame(data=self.data[:, 0, :], index=self.trade_days, columns=self.codes)
        else:
            return pd.DataFrame(data=self.data[:, self.cols[field], :], index=self.trade_days, columns=self.codes)



if __name__ == '__main__':
    test = BaseDataLoader.load_data('./数据/stock_bar_1day.parquet', codes=['000001.SZ', '000002.SZ'])
    x = test.to_dataframes()
    y = 1