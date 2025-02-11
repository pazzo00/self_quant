import bisect
from itertools import product, chain
import pandas as pd
from warnings import warn
import numpy as np
import multiprocessing as mp
from base_data_loader import BaseDataLoader
from matplotlib import pyplot as plt
from functools import partial



class SignalAnalyzer:
    def __init__(self, base_data_info, freq, offset, bins=5):
        assert 'signal_data' in base_data_info, 'base_data_info中必须包含signal_data'
        assert 'ret_data' in base_data_info, 'base_data_info中必须包含ret_data'
        self.load_data(base_data_info['signal_data'], 'signal_data')
        for k, v in base_data_info.items():
            if k != 'signal_data':
                v['codes'] = self.__getattribute__('signal_data').codes
                self.load_data(v, k)
        self.freq = freq
        self.offset = offset
        self.bins = bins
        self.x = self.y = None
        self.ind_unique = self.ind_numerical = None
        self.ret_df = self.signal_df = None
        self.ret_1d = self.weights = self.balance_idx = None

    def load_data(self, data, name):
        if isinstance(data, pd.DataFrame):
            if 'code' not in data.columns:
                assert 'datetime' in str(data.index.dtype), 'signal_data的索引不是datetime类型'
                data.reset_index(inplace=True)
                data = pd.melt(data, id_vars='datetime', var_name='code', value_name='signal')
                data = BaseDataLoader.from_dataframe(data)
            else:
                data = BaseDataLoader.from_dataframe(data)
        elif isinstance(data, BaseDataLoader):
            data = data
        elif isinstance(data, dict):
            data = BaseDataLoader.load_data(**data)
        else:
            warn('输入的data类型暂时不支持')
        self.__setattr__(name, data)

    def pre_transform(self, cap_data=None, cap_name=None, ind_data=None, ind_name=None):
        self.remove_outliers()
        self.neutralization(cap_data, cap_name, ind_data, ind_name)
        self.align()

    def remove_outliers(self):
        signal_data = self.__getattribute__('signal_data')
        median = np.nanmedian(signal_data.data, axis=2)
        absolute_deviation = np.abs(signal_data.data - median[:, :, np.newaxis])
        mad = np.nanmedian(absolute_deviation, axis=2)
        limit_up = median + 5 * mad
        limit_down = median - 5 * mad
        signal_data.data = np.clip(signal_data.data, limit_down[:, :, np.newaxis], limit_up[:, :, np.newaxis])

    def neutralization(self, cap_data, cap_name, ind_data, ind_name):
        if (cap_data is None and ind_data is None) or (not (hasattr(self, cap_data) and
                                                            cap_name in self.__getattribute__(cap_data).cols) and
                (not hasattr(self, ind_data) and ind_name in self.__getattribute__(ind_data).cols)):
            warn('缺少市值数据和行业数据，无法进行行业市值中性化')
            return
        signal_data = self.__getattribute__('signal_data')
        self.y = signal_data.data[:, 0, :]

        if ind_data is None:
            warn('缺少行业数据，无法进行行业中性化')
            x = []
        else:
            self.ind_data_numerical(ind_data, ind_name)
            x = self.ind_numerical[:]

        if hasattr(self, cap_data) and cap_name in self.__getattribute__(cap_data).cols:
            cap_data = self.__getattribute__('market_cap')
            x.append(np.log(cap_data.data[:, cap_data.cols[cap_name], np.newaxis, :]))
        else:
            warn('缺少市值数据，无法进行市值中性化')

        self.x = np.concatenate(x, axis=1)
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     res = pool.map(self.cal_residual, list(range(len(self.y))))
        res = []
        for i in range(len(self.y)):
            res.append(self.cal_residual(i))
        res = np.array(res)
        signal_data.data[:, 0, :] = res

    def ind_data_numerical(self, ind_data, ind_name):
        if self.ind_numerical is not None:
            return

        if hasattr(self, ind_data) and ind_name in self.__getattribute__(ind_data).cols:
            self.ind_unique = self.__getattribute__('ind_data')
            self.ind_unique = self.ind_unique.data[:, self.ind_unique.cols[ind_name], :].astype('<U32')
            self.ind_numerical = self.ind_unique[:, np.newaxis, :]
            self.ind_unique = np.unique(self.ind_unique[~np.isin(self.ind_unique, ['None', np.nan])])
            x = []
            for tag in self.ind_unique:
                x.append(np.nan_to_num(self.ind_numerical == tag, nan=0))
            self.ind_numerical = x

    @staticmethod
    def __regression__(x, y):
        if len(x.shape) == 2:
            flag = ~np.isnan(x[:, -1]) & ~np.isnan(y)
            x_data, y_data = x[flag, :], y[flag]
        else:
            flag = ~np.isnan(x) & ~np.isnan(y)
            x_data, y_data = x[flag], y[flag]
        x_b = np.c_[np.ones((x_data.shape[0], 1)), x_data]
        w = np.linalg.pinv(x_b.T @ x_b) @ x_b.T @ y_data
        return w

    def cal_residual(self, idx):
        x, y = self.x[idx, :, :].T, self.y[idx, :]
        w = self.__regression__(x, y)
        return y - w @ np.c_[np.ones((x.shape[0], 1)), x].T

    def align(self):
        signal_data = self.__getattribute__('signal_data')
        if self.offset == 'close':
            # 收盘时候计算的因子，只能在第二天使用
            signal_data.data = signal_data.data[:-1]
            signal_data.trade_days = signal_data.trade_days[1:]

    def get_ret_and_signal(self):
        if self.signal_df is None:
            signal_df = self.__getattribute__('signal_data').to_dataframe()
            ret_df = self.__getattribute__('ret_data').to_dataframe()
            ret_df = ret_df.shift(-self.freq) / ret_df - 1
            ret_df = ret_df.dropna(how='all')
            self.signal_df = signal_df
            self.ret_df = ret_df

    def ic_analysis(self, ind_data=None, ind_name=None):
        self.get_ret_and_signal()
        rank_ic = self.signal_df.reindex(self.ret_df.index).corrwith(self.ret_df, method='spearman', axis=1)

        args = [(rank_ic, '')]
        if ind_data is None:
            warn('缺少行业数据，不进行行业IC分析')
        else:
            self.ind_data_numerical(ind_data, ind_name)
            if self.ind_numerical is not None:
                for i, _ in enumerate(self.ind_unique):
                    tmp = self.ind_numerical[i][1:, 0, :]
                    tmp = np.where(tmp, 1.0, np.nan)
                    tmp = (self.signal_df * tmp).reindex(self.ret_df.index).corrwith(
                        self.ret_df, method='spearman', axis=1)
                    args.append((tmp, _))

        if len(args) == 1:
            self.__ic_analysis__(args[0])
        else:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                pool.map(self.__ic_analysis__, args)

    def __ic_analysis__(self, args):
        rank_ic, ind = args
        self.ic_group_analysis(rank_ic, 'YE', ind)
        self.ic_group_analysis(rank_ic, 'ME', ind)

        if ind == '':
            rank_ic = pd.DataFrame(rank_ic, columns=['rank_ic'])
            rank_ic['rank_ic_{}mean'.format(self.freq)] = rank_ic['rank_ic'].rolling(self.freq).mean()
            rank_ic.index = rank_ic.index.date
            self.plot(rank_ic, 'IC', 'time', ind, mode='line')

            # 不能开进程池了，可以换个方式继续多进程，然后测试一下多进程套用多进程的代码
            ic_cumsum = []
            for i in range(self.freq):
                ic_cumsum.append(self.split_by_freq(rank_ic, 0, i))
            ic_cumsum = pd.concat(ic_cumsum, axis=1)
            ic_cumsum = ic_cumsum.fillna(0)
            ic_cumsum = ic_cumsum.cumsum()
            max_idx = np.nanargmax(ic_cumsum.iloc[-1])
            min_idx = np.nanargmin(ic_cumsum.iloc[-1])
            ic_cumsum = ic_cumsum.iloc[:, [max_idx, min_idx]]
            ic_cumsum.columns = ['max_cumsum', 'min_cumsum']
            self.plot(ic_cumsum, 'IC累加', 'day', ind, mode='line')

    def ic_group_analysis(self, data, freq, ind):
        grouper = [pd.Grouper(freq=freq)]
        gp_data = data.groupby(grouper).agg(ic_mean='mean', ic_std='std')
        gp_data['ic_ir'] = gp_data.eval('ic_mean / ic_std')
        if freq == 'YE':
            gp_data.index = gp_data.index.year
            gp_data = gp_data.dropna()
            x_name = 'year'
        else:
            gp_data['year'] = gp_data.index.year
            gp_data['month'] = gp_data.index.month
            gp_data = {'ic_mean': pd.pivot_table(gp_data, index='month', columns='year', values='ic_mean'),
                       'ic_ir': pd.pivot_table(gp_data, index='month', columns='year', values='ic_ir')}
            x_name = 'month'
        self.plot(gp_data['ic_mean'], x_name + '_ic_mean', x_name, ind, mode='bar')
        self.plot(gp_data['ic_ir'], x_name + '_ic_ir', x_name, ind, mode='bar')

    def split_by_freq(self, data, idx_2, idx_1):
        tmp = data.iloc[list(range(idx_1, len(data), self.freq)), idx_2]
        tmp.reset_index(inplace=True, drop=True)
        return tmp

    def regression_analysis(self):
        self.get_ret_and_signal()
        self.x = self.signal_df.values
        y = self.ret_df.reindex(self.signal_df.index).dropna(how='all')
        time_index = y.index
        self.y = y.values
        # # 多进程
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     res = pool.map(self.__regression_analysis__, list(range(len(self.y))))

        res = []
        for i in range(len(self.y)):
            res.append(self.__regression_analysis__(i))
        factor_ret = pd.DataFrame(data=np.array(res), index=time_index, columns=['factor_return'])
        factor_ret['factor_return_{}ma'.format(self.freq)] = factor_ret['factor_return'].rolling(self.freq).mean()

        self.plot(factor_ret, 'factor_return', 'time', mode='line')

        ret_cumprod = []
        for i in range(self.freq):
            ret_cumprod.append(self.split_by_freq(factor_ret, 0, i))
        ret_cumprod = pd.concat(ret_cumprod, axis=1)
        ret_cumprod = (1 + ret_cumprod).cumprod()
        ret_cumprod = ret_cumprod.dropna()
        max_idx = np.nanargmax(ret_cumprod.iloc[-1])
        min_idx = np.nanargmin(ret_cumprod.iloc[-1])
        ic_cumsum = ret_cumprod.iloc[:, [max_idx, min_idx]]
        ic_cumsum.columns = ['max_cumsum', 'min_cumsum']
        self.plot(ic_cumsum, '因子收益率累加', 'day', mode='line')

    def __regression_analysis__(self, idx):
        x, y = self.x[idx, :], self.y[idx, :]
        x = (x - np.nanmean(x)) / np.nanstd(x)
        w = self.__regression__(x, y)
        return w[-1]

    def return_analysis(self, weights_mode, data_name=None):
        assert weights_mode == 'equal' or (
                hasattr(self, data_name) and weights_mode in self.__getattribute__(data_name).cols.keys()), \
            '输入的权重不合法'

        self.ret_1d = self.__getattribute__('ret_data').to_dataframe()
        self.ret_1d = self.ret_1d / self.ret_1d.shift(1) - 1
        self.ret_1d = self.ret_1d.dropna(how='all')

        self.get_ret_and_signal()
        self.cal_weights(weights_mode, data_name)

        start = self.ret_1d.index.tolist().index(self.signal_df.iloc[[0]].index)
        self.ret_1d = self.ret_1d.iloc[start:]
        # # 多进程
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     group_ret = pool.map(self.cal_group_ret, list(range(1, len(self.ret_1d))))

        group_ret = []
        for i in range(1, len(self.ret_1d), 1):
            group_ret.append(self.cal_group_ret(i))
        group_ret = [[0] * self.bins] + group_ret
        group_ret = pd.DataFrame(data=group_ret, index=self.ret_1d.index, columns=self.weights.keys())
        group_ret.to_csv('./ret.csv')
        group_ret = (1 + group_ret).cumprod()

        self.plot(group_ret, '分层回测', x_name='time', mode='line')

    def cal_weights(self, weights_mode, data_name):
        if self.weights is None:
            self.balance_idx = list(range(0, len(self.signal_df), self.freq))
            # # 多进程
            # with mp.Pool(processes=mp.cpu_count()) as pool:
            #     weights = pool.map(partial(self.__cal_weights__, weights_mode, data_name), self.balance_idx)

            weights = []
            for idx in self.balance_idx:
                weights.append(self.__cal_weights__(weights_mode, data_name, idx))
            self.weights = {}
            for i in range(self.bins):
                self.weights['group_{}'.format(i)] = pd.concat([row[i] for row in weights]).groupby('index')

    def __cal_weights__(self, weights_mode, data_name, idx):
        signal = self.signal_df.iloc[[idx]].T
        signal = signal.dropna()
        day = signal.columns[0]
        signal.columns = ['signal']
        if weights_mode != 'equal':
            weights_df = self.__getattribute__(data_name).get_window_df(weights_mode, 0, day).T
            weights_df.columns = ['weight']
            signal = signal.join(weights_df)
        signal.sort_values(by='signal', inplace=True)

        gap = len(signal) // self.bins

        weights = []
        for i in range(self.bins):
            if i != self.bins - 1:
                tmp = signal.iloc[i * gap: (i + 1) * gap]
            else:
                tmp = signal.iloc[i * gap:]
            if weights_mode == 'equal':
                tmp['weight'] = 1 / len(tmp)
            else:
                tmp['weight'] = tmp['weight'] / tmp['weight'].sum()
            tmp = tmp[['weight']]
            tmp.index.name = 'code'
            tmp['index'] = idx
            tmp['datetime'] = day
            weights.append(tmp)
        return weights

    def cal_group_ret(self, idx):
        data = self.ret_1d.iloc[idx]
        balance_id = bisect.bisect_left(self.balance_idx, idx) - 1
        res = []
        for v in self.weights.values():
            v = v.get_group(self.balance_idx[balance_id])
            tmp = v[['weight']].join(data)
            tmp = tmp.fillna(0)
            res.append(tmp.iloc[:, 0].dot(tmp.iloc[:, 1]))
        return res

    def turnover_analysis(self, weights_mode, data_name=None, ind_data=None, ind_name=None):
        self.get_ret_and_signal()
        self.cal_weights(weights_mode, data_name)

        # # 多进程
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     group_turnover = pool.map(self.__turnover_analysis__, list(range(1, len(self.balance_idx))))

        group_turnover = []
        for i in range(1, len(self.balance_idx)):
            group_turnover.append(self.__turnover_analysis__(i))
        group_turnover = pd.Series(data=np.nanmean(group_turnover, axis=0), index=self.weights.keys())
        self.plot(group_turnover, '每次调仓的平均换手率', x_name='', mode='bar')

        if ind_data is None or (
                not (hasattr(self, ind_data) and ind_name in self.__getattribute__(ind_data).cols)):
            warn('没有行业板块数据，不进行行业板块换手率分析')
            return
        group_ind_turnover = []
        for i in range(1, len(self.balance_idx)):
            group_ind_turnover.append(self.__ind_turnover_analysis__(ind_data, ind_name, i))

        # # 多进程
        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     group_ind_turnover = pool.map(partial(self.__ind_turnover_analysis__, ind_data, ind_name),
        #                                   list(range(1, len(self.balance_idx))))

        group_ind_turnover = pd.Series(data=np.nanmean(group_ind_turnover, axis=0), index=self.weights.keys())
        self.plot(group_ind_turnover, '每次调仓的平均行业换手率', x_name='', mode='bar')

    def __turnover_analysis__(self, idx, last_ind=None, ind=None):
        res = []
        for key in self.weights:
            data = self.weights[key]
            last_weights = data.get_group(self.balance_idx[idx - 1])[['weight']]
            last_weights.rename(columns={'weight': 'last_weight'}, inplace=True)
            weights = data.get_group(self.balance_idx[idx])[['weight']]
            if last_ind is not None:
                last_weights = last_weights.join(last_ind).groupby('ind_name').sum()
                weights = weights.join(ind).groupby('ind_name').sum()
            weights = weights.join(last_weights, how='outer')
            weights = weights.fillna(0)
            res.append(np.sum(np.abs(weights.iloc[:, 0] - weights.iloc[:, 1])) / 2)
        return res

    def __ind_turnover_analysis__(self, ind_data, ind_name, idx):
        last_ind = self.get_ind_data_by_day(ind_data, ind_name, self.signal_df.index[self.balance_idx[idx - 1]])
        ind = self.get_ind_data_by_day(ind_data, ind_name, self.signal_df.index[self.balance_idx[idx]])
        return self.__turnover_analysis__(idx, last_ind, ind)

    def get_ind_data_by_day(self, ind_data, ind_name, day):
        data = self.__getattribute__(ind_data).get_window_df(ind_name, 0, day).T
        data.columns = ['ind_name']
        return data

    def attribution_analysis(self, weights_mode, factor_data, data_name=None):
        self.get_ret_and_signal()
        self.cal_weights(weights_mode, data_name)
        data_info, fields = self.dict_product(factor_data)

        group_attribution = []
        for arg in data_info:
            # # 多进程
            # with mp.Pool(processes=4) as pool:
            #     res = pool.map(partial(self.__attribution_analysis__, arg[0], arg[1]), self.balance_idx)

            res = []
            for i in self.balance_idx:
                res.append(self.__attribution_analysis__(arg[0], arg[1], i))
            group_attribution.append(res)
        group_attribution = pd.DataFrame(data=np.nanmean(group_attribution, axis=1), index=fields,
                                         columns=self.weights.keys())
        self.plot(group_attribution, '每组归因分析', x_name='因子名', mode='bar', rotation=0)

    def __attribution_analysis__(self, data_name, field_name, idx):
        res = []
        day = self.signal_df.index[idx]
        data = self.__getattribute__(data_name).get_window_df(field_name, 0, day)
        data = data.sub(data.mean(axis=1), axis=0).div(data.std(axis=1), axis=0).T
        for key in self.weights:
            weights = self.weights[key].get_group(idx)[['weight']]
            weights = weights.join(data)
            weights = weights.dropna()
            res.append(weights.iloc[:, 0].dot(weights.iloc[:, 1]))
        return res

    # 嵌套多进程，速度变慢
    # def attribution_analysis(self, weights_mode, factor_data, data_name=None):
    #     self.get_ret_and_signal()
    #     self.cal_weights(weights_mode, data_name)
    #     data_info, fields = self.dict_product(factor_data)
    #
    #     queue = Queue()
    #     processes = []
    #     for arg in data_info:
    #         process = Process(target=self.__attribution_analysis__, args=(arg, queue))
    #         processes.append(process)
    #         process.start()
    #
    #     for process in processes:
    #         process.join()
    #     group_attribution = [queue.get() for _ in range(len(data_info))]
    #
    #     group_attribution = pd.DataFrame(data=np.nanmean(group_attribution, axis=1), index=fields,
    #                                   columns=self.weights.keys())
    #     self.plot(group_attribution, '每组归因分析', x_name='因子名', mode='bar', rotation=0)
    #
    # def __attribution_analysis__(self, args, queue):
    #     new_queue = Queue()
    #     processes = []
    #     for idx in self.balance_idx:
    #         process = Process(target=self.__attribution_analysis_by_day__, args=(idx, args[0], args[1], new_queue))
    #         processes.append(process)
    #         process.start()
    #
    #     for process in processes:
    #         process.join()
    #     queue.put([new_queue.get() for _ in range(len(self.balance_idx))])
    #
    # def __attribution_analysis_by_day__(self, idx, data_name, field_name, queue):
    #     res = []
    #     day = self.signal_df.index[idx]
    #     data = self.__getattribute__(data_name).get_window_df(field_name, 0, day)
    #     data = data.sub(data.mean(axis=1), axis=0).div(data.std(axis=1), axis=0).T
    #     for key in self.weights:
    #         weights = self.weights[key].get_group(idx)[['weight']]
    #         weights = weights.join(data)
    #         weights = weights.dropna()
    #         res.append(weights.iloc[:, 0].dot(weights.iloc[:, 1]))
    #     queue.put(res)

    def dict_product(self, data_info_dict):
        data_info = []
        for k, v in data_info_dict.items():
            data_info.append(self.__dict_product__((k, v)))
        data_info = list(chain.from_iterable(data_info))
        return data_info, np.array(data_info)[:, 1]

    @staticmethod
    def __dict_product__(args):
        k, v = [args[0]], args[1]
        return list(product(*[k, v]))

    def corr_analysis(self, data_name):
        self.get_ret_and_signal()
        data_info, fields = self.dict_product(data_name)
        corr_res = []
        for info in data_info:
            corr_res.append(self.__corr_analysis__(info))

        # # 多进程
        # with mp.Pool(processes=4) as pool:
        #     corr_res = pool.map(self.__corr_analysis__, data_info)
        corr_res = pd.Series(data=corr_res, index=fields)
        self.plot(corr_res, '相关性分析', '因子名', mode='bar')

    def __corr_analysis__(self, args):
        data = self.__getattribute__(args[0]).to_dataframe(args[1])
        test = self.signal_df.reindex(data.index)
        return np.nanmean(test.corrwith(data, method='spearman', axis=1))

    @staticmethod
    def plot(data, title, x_name, ind='', mode='bar', rotation=None):
        plt.figure()
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        if mode == 'bar':
            data.plot(kind='bar')
        elif mode == 'line':
            data.plot()
        plt.title(title)
        plt.xlabel(x_name)
        if rotation is not None:
            plt.xticks(rotation=rotation)
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(title.replace(' ', '_') + '_' + ind + '.png', format='png', dpi=1000)


if __name__ == '__main__':
    data_dict = {
        'signal_data': {'path': './数据/stock_bar_1day.parquet',
                        'start': '2018-01-01',
                        'end': '2021-01-01', 'fields': ['volume']},
        'ret_data': {'path': './数据/stock_bar_1day.parquet', 'fields': ['close'], 'start': '2018-01-01',
                     'end': '2021-03-01'},
        'market_cap': {'path': './数据/capital.parquet',
                       'fields': ['market_cap', 'circulating_cap', 'pe_ratio', 'turnover_ratio', 'pb_ratio'],
                       'start': '2018-01-01', 'end': '2021-01-01'},
        'ind_data': {'path': './数据/sw_industry.parquet', 'fields': ['sw_l1_code'], 'start': '2018-01-01',
                     'end': '2021-01-01'},
        'factor_data': {'path': './数据/stock_bar_1day.parquet', 'fields': ['turnover', 'volume'],
                        'start': '2018-01-01', 'end': '2021-01-01'}
    }
    sa = SignalAnalyzer(data_dict, freq=21, offset='close')
    sa.pre_transform('market_cap', 'market_cap', 'ind_data', 'sw_l1_code')
    sa.ic_analysis('ind_data', 'sw_l1_code')
    # sa.regression_analysis()
    # sa.return_analysis('equal')
    # sa.turnover_analysis('equal', ind_data='ind_data', ind_name='sw_l1_code')
    # sa.attribution_analysis('equal',
    #                         {'market_cap': ['market_cap', 'circulating_cap', 'turnover_ratio', 'pe_ratio', 'pb_ratio'],
    #                          'factor_data': ['turnover', 'volume']})
    # sa.corr_analysis({'market_cap': ['market_cap', 'circulating_cap', 'turnover_ratio', 'pe_ratio', 'pb_ratio'],
    #                          'factor_data': ['turnover', 'volume']})